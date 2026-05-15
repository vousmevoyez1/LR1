"""
Jina V4oa compatible thought-initialization embedder.

Loads LoRA checkpoint while skipping embedding weights, then re-initializes
reasoning tokens (<thought_i> and <final>) from a specified semantic token.
This avoids changing original common scripts.
"""

import os
import argparse

import torch
from omegaconf import OmegaConf

import dist_utils
from utils import build_model_from_config as base_build_model_from_config, set_seed
from mbeir_embedder import generate_embeds_for_config


def build_model_from_config_v4oa(config):
    try:
        return base_build_model_from_config(config)
    except NotImplementedError:
        model_name = config.model.name
        if model_name != "JinaEmbeddingsV4oaModel":
            raise

        from models.jina_v4oa.jina_v4oa.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model

        model_config = config.model
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_config.original_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        model.mbeir_task_label = getattr(model_config, "mbeir_task_label", "retrieval")
        model.mbeir_image_size = tuple(map(int, config.data_config.image_size.split(",")))
        model.mbeir_max_text_length = getattr(model_config, "mbeir_max_text_length", 512)
        model.mbeir_num_thought_tokens = getattr(model_config, "mbeir_num_thought_tokens", 0)

        if model.mbeir_num_thought_tokens > 0:
            model.setup_thought_tokens(model.mbeir_num_thought_tokens, skip_init=True)

        model.task = model.mbeir_task_label
        return model


def load_lora_checkpoint_skip_embed(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"Checkpoint file {checkpoint_path} does not exist")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "lora_adapter" in checkpoint:
        adapter_state_dict = checkpoint["lora_adapter"]
        model_state_dict = model.state_dict()
        lora_count = 0
        embed_skipped = 0

        for name, param in adapter_state_dict.items():
            if name in model_state_dict:
                if "embed_tokens" in name:
                    embed_skipped += 1
                    continue
                model_state_dict[name].copy_(param)
                if "lora_" in name.lower():
                    lora_count += 1
            else:
                print(f"Warning: parameter {name} not found in model")

        print(f"Loaded from {checkpoint_path} (thought_init mode):")
        print(f"  - LoRA parameters loaded: {lora_count}")
        print(f"  - Embedding parameters SKIPPED: {embed_skipped}")

    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        model_state_dict = model.state_dict()
        lora_keys_loaded = 0
        embed_skipped = 0

        for name, param in state_dict.items():
            if name in model_state_dict:
                if "lora_" in name.lower():
                    model_state_dict[name].copy_(param)
                    lora_keys_loaded += 1
                elif "embed_tokens" in name:
                    embed_skipped += 1
                    continue

        print(f"Loaded {lora_keys_loaded} LoRA parameters from {checkpoint_path}")
        print(f"Skipped {embed_skipped} embedding parameters (thought_init mode)")
    else:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no 'lora_adapter' or 'model' key")

    return model, checkpoint


def reinit_reasoning_tokens_from_token(model, init_token=".", epsilon=0.01):
    """
    Re-initialize all reasoning token embeddings (<thought_i>, <final>) from
    semantic token embedding + Gaussian perturbation.
    """
    tokenizer = model.processor.tokenizer
    embed_layer = model.model.language_model.embed_tokens

    token_ids = tokenizer.encode(init_token, add_special_tokens=False)
    if len(token_ids) == 0:
        raise ValueError(f"Cannot encode init token: {init_token}")

    base_token_id = token_ids[0]
    if base_token_id >= embed_layer.weight.shape[0]:
        raise ValueError(f"Init token id out of range: {base_token_id}")

    base_embedding = embed_layer.weight.data[base_token_id].clone()

    reasoning_token_ids = []
    if getattr(model, "_thought_token_ids", None) is not None:
        reasoning_token_ids.extend(model._thought_token_ids)
    if getattr(model, "_final_token_id", None) is not None:
        reasoning_token_ids.append(model._final_token_id)

    if len(reasoning_token_ids) == 0:
        print("No reasoning token ids found; skip re-initialization")
        return

    with torch.no_grad():
        for token_id in reasoning_token_ids:
            perturbation = torch.randn_like(base_embedding) * epsilon
            embed_layer.weight.data[token_id] = base_embedding + perturbation

    print(
        f"Re-initialized {len(reasoning_token_ids)} reasoning tokens from '{init_token}' "
        f"with epsilon={epsilon}"
    )


def main(config, lora_checkpoint=None, thought_init_token="."):
    seed = config.seed + dist_utils.get_rank()
    set_seed(seed)

    model = build_model_from_config_v4oa(config)

    if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        if dist_utils.is_main_process():
            print(f"Loading LoRA checkpoint (skip embed): {lora_checkpoint}")
        model, _ = load_lora_checkpoint_skip_embed(model, lora_checkpoint)
        if dist_utils.is_main_process():
            print("LoRA checkpoint loaded (embed_tokens skipped)!")

    num_thought_tokens = getattr(model, "mbeir_num_thought_tokens", 0)
    if num_thought_tokens > 0:
        if dist_utils.is_main_process():
            print(f"\n{'=' * 60}")
            print(f"Re-initializing reasoning tokens to '{thought_init_token}'")
            print(f"{'=' * 60}")
        reinit_reasoning_tokens_from_token(model, init_token=thought_init_token, epsilon=0.01)

    model.eval()

    if not callable(getattr(model, "encode_mbeir_batch")):
        raise AttributeError("The provided model does not have a callable 'encode_mbeir_batch' method.")
    if not callable(getattr(model, "get_img_preprocess_fn")):
        raise AttributeError("The provided model does not have an 'img_preprocess_fn' attribute.")
    if not callable(getattr(model, "get_tokenizer")):
        raise AttributeError("The provided model does not have a 'tokenizer' attribute.")

    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()

    model = model.to(config.dist_config.gpu_id)
    print(f"Model is set up on GPU {config.dist_config.gpu_id}.")

    generate_embeds_for_config(
        model=model,
        img_preprocess_fn=img_preprocess_fn,
        tokenizer=tokenizer,
        config=config,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="MBEIR Embedder with thought token re-initialization (Jina V4oa)")
    parser.add_argument("--uniir_dir", type=str, default="/data/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="/data/UniIR/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument("--reason_steps", type=int, default=0, help="(Deprecated) Number of implicit reasoning steps")
    parser.add_argument(
        "--num_thought_tokens",
        type=int,
        default=None,
        help="Number of thought tokens for reasoning (overrides config)",
    )
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument(
        "--thought_init_token",
        type=str,
        default=".",
        help="Token to use for re-initializing reasoning token embeddings (default: '.')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_arguments()
    config = OmegaConf.load(args.config_path)

    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    if not hasattr(config, "model"):
        config.model = OmegaConf.create({})
    config.model.mbeir_reason_steps = args.reason_steps

    if args.num_thought_tokens is not None:
        config.model.mbeir_num_thought_tokens = args.num_thought_tokens

    steps_for_path = args.num_thought_tokens if args.num_thought_tokens is not None else args.reason_steps
    if steps_for_path > 0:
        original_path_suffix = config.experiment.path_suffix
        config.experiment.path_suffix = f"{original_path_suffix}reason_steps_{steps_for_path}/"

    args.dist_url = config.dist_config.dist_url
    dist_utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    if dist_utils.is_main_process():
        print(f"\n{'=' * 60}")
        print("Thought Token Init Test (Jina V4oa)")
        print(f"  num_thought_tokens: {args.num_thought_tokens}")
        print(f"  thought_init_token: '{args.thought_init_token}'")
        print(f"  lora_checkpoint: {args.lora_checkpoint}")
        print(f"  Output path_suffix: {config.experiment.path_suffix}")
        print(f"{'=' * 60}\n")
        print(OmegaConf.to_yaml(config, sort_keys=False))

    main(config, lora_checkpoint=args.lora_checkpoint, thought_init_token=args.thought_init_token)

    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
