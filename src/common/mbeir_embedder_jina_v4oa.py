"""
Jina V4oa compatible embedder entrypoint.

This file is a non-invasive wrapper for /data/LR1/src/common/mbeir_embedder.py
that adds support for model name `JinaEmbeddingsV4oaModel` without modifying
original common files.
"""

import os
import argparse

import torch
from omegaconf import OmegaConf

import dist_utils
import mbeir_embedder as base_embedder
from utils import build_model_from_config as base_build_model_from_config, set_seed


def build_model_from_config_v4oa(config):
    """Build model with fallback support for JinaEmbeddingsV4oaModel."""
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
            # Keep interface parity with existing embedder logic.
            model.setup_thought_tokens(model.mbeir_num_thought_tokens, skip_init=True)

        model.task = model.mbeir_task_label
        return model


def main(config, lora_checkpoint=None):
    seed = config.seed + dist_utils.get_rank()
    set_seed(seed)

    model = build_model_from_config_v4oa(config)

    if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        if dist_utils.is_main_process():
            print(f"Loading LoRA checkpoint from: {lora_checkpoint}")

        model_name = config.model.name
        if model_name == "JinaEmbeddingsV4oModel":
            from models.jina_v4o.train import load_lora_checkpoint

            model, _ = load_lora_checkpoint(model, lora_checkpoint)
        elif model_name == "JinaEmbeddingsV4oaModel":
            from models.jina_v4oa.train import load_lora_checkpoint

            model, _ = load_lora_checkpoint(model, lora_checkpoint)
        else:
            from models.jina_v4.train import load_lora_checkpoint, freeze_base_model_keep_lora

            model = freeze_base_model_keep_lora(model)
            model, _ = load_lora_checkpoint(model, lora_checkpoint)

        if dist_utils.is_main_process():
            print("Checkpoint loaded successfully!")

    model.eval()

    if not callable(getattr(model, "encode_mbeir_batch")):
        raise AttributeError("The provided model does not have a callable 'encode' method.")
    if not callable(getattr(model, "get_img_preprocess_fn")):
        raise AttributeError("The provided model does not have an 'img_preprocess_fn' attribute.")
    if not callable(getattr(model, "get_tokenizer")):
        raise AttributeError("The provided model does not have a 'tokenizer' attribute.")

    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()

    model = model.to(config.dist_config.gpu_id)
    print(f"Model is set up on GPU {config.dist_config.gpu_id}.")

    base_embedder.generate_embeds_for_config(
        model=model,
        img_preprocess_fn=img_preprocess_fn,
        tokenizer=tokenizer,
        config=config,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Embeddings for MBEIR (Jina V4oa)")
    parser.add_argument("--uniir_dir", type=str, default="/data/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="/data/UniIR/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--reason_steps",
        type=int,
        default=0,
        help="(Deprecated) Number of implicit reasoning steps in embedding space (0 = disabled)",
    )
    parser.add_argument(
        "--num_thought_tokens",
        type=int,
        default=None,
        help="Number of thought tokens for reasoning (overrides config if set)",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA checkpoint to load (optional)",
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
        if args.num_thought_tokens is not None:
            print(f"Running with num_thought_tokens={args.num_thought_tokens}")
        else:
            print(f"Running with reason_steps={args.reason_steps}")
        if args.lora_checkpoint:
            print(f"LoRA checkpoint: {args.lora_checkpoint}")
        print(f"Output path_suffix: {config.experiment.path_suffix}")
        print(OmegaConf.to_yaml(config, sort_keys=False))

    main(config, lora_checkpoint=args.lora_checkpoint)

    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
