"""
Modified MBEIR embedder that loads LoRA checkpoint but RE-INITIALIZES 
thought token embeddings to a specified token (e.g., ".") instead of 
loading them from the checkpoint.

This is used to test the effect of thought token initialization on retrieval
performance, isolating the contribution of learned thought token embeddings.

Usage:
    python -m torch.distributed.run \
        --nproc_per_node=2 \
        mbeir_embedder_thought_init.py \
        --config_path <embed.yaml> \
        --uniir_dir <output_dir> \
        --mbeir_data_dir <data_dir> \
        --num_thought_tokens 5 \
        --lora_checkpoint <checkpoint.pth> \
        --thought_init_token "."
"""

import os
import argparse
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import dist_utils
from utils import build_model_from_config, set_seed
from mbeir_embedder import generate_embeds_for_config


def load_lora_checkpoint_skip_embed(model, checkpoint_path):
    """
    Load LoRA adapter weights from checkpoint, but SKIP loading 
    thought token embeddings (embed_tokens). 
    
    This allows us to test thought tokens with fresh initialization
    while still using the trained LoRA weights.
    
    Args:
        model: The model with thought tokens already set up
        checkpoint_path: Path to the LoRA checkpoint
        
    Returns:
        model: Model with loaded LoRA weights (but original thought token embeddings)
        checkpoint: Full checkpoint dict
    """
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
                # Skip embedding parameters (thought token embeddings)
                if 'embed_tokens' in name:
                    embed_skipped += 1
                    continue
                model_state_dict[name].copy_(param)
                if 'lora_' in name.lower():
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
                if 'lora_' in name.lower():
                    model_state_dict[name].copy_(param)
                    lora_keys_loaded += 1
                elif 'embed_tokens' in name:
                    # Skip embedding parameters
                    embed_skipped += 1
                    continue
        
        print(f"Loaded {lora_keys_loaded} LoRA parameters from {checkpoint_path}")
        print(f"Skipped {embed_skipped} embedding parameters (thought_init mode)")
    else:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no 'lora_adapter' or 'model' key")
    
    return model, checkpoint


def main(config, lora_checkpoint=None, thought_init_token="."):
    """
    Main function: build model, load LoRA (skip embed), re-init thought tokens, embed.
    """
    seed = config.seed + dist_utils.get_rank()
    set_seed(seed)

    # Step 1: Build model (with skip_init=True as usual)
    model = build_model_from_config(config)
    
    # Step 2: Load LoRA checkpoint but SKIP thought token embeddings
    if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        if dist_utils.is_main_process():
            print(f"Loading LoRA checkpoint (skip embed): {lora_checkpoint}")
        model, _ = load_lora_checkpoint_skip_embed(model, lora_checkpoint)
        if dist_utils.is_main_process():
            print("LoRA checkpoint loaded (embed_tokens skipped)!")
    
    # Step 3: Re-initialize thought token embeddings to the specified token
    num_thought_tokens = getattr(model, 'mbeir_num_thought_tokens', 0)
    if num_thought_tokens > 0:
        if dist_utils.is_main_process():
            print(f"\n{'='*60}")
            print(f"Re-initializing {num_thought_tokens} thought tokens to '{thought_init_token}'")
            print(f"{'='*60}")
        # Call semantic init with epsilon=0.01 (same as training default)
        model._semantic_init_thought_tokens(
            semantic_init_token=thought_init_token, 
            epsilon=0.01
        )
        if dist_utils.is_main_process():
            print(f"Thought tokens re-initialized to '{thought_init_token}' with epsilon=0.01")
    
    model.eval()
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    
    # 强制写入属性，让 encode_mbeir_batch 能够读到 5
    base_model.mbeir_num_thought_tokens = config.model.mbeir_num_thought_tokens
    base_model.mbeir_thought_pooling_mode = "last"
    # Step 4: Generate embeddings
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
    parser = argparse.ArgumentParser(
        description="MBEIR Embedder with thought token re-initialization"
    )
    parser.add_argument("--uniir_dir", type=str, default="/data/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="/data/UniIR/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument("--reason_steps", type=int, default=0,
                        help="(Deprecated) Number of implicit reasoning steps")
    parser.add_argument("--num_thought_tokens", type=int, default=None,
                        help="Number of thought tokens for reasoning (overrides config)")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--thought_init_token", type=str, default=".",
                        help="Token to use for re-initializing thought token embeddings (default: '.')")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_arguments()
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    if not hasattr(config, 'model'):
        config.model = OmegaConf.create({})
    config.model.mbeir_reason_steps = args.reason_steps
    
    if args.num_thought_tokens is not None:
        config.model.mbeir_num_thought_tokens = args.num_thought_tokens
    
    steps_for_path = args.num_thought_tokens if args.num_thought_tokens is not None else args.reason_steps
    
    if steps_for_path > 0:
        original_path_suffix = config.experiment.path_suffix
        config.experiment.path_suffix = f"{original_path_suffix}reason_steps_{steps_for_path}/"

    # Initialize distributed mode
    args.dist_url = config.dist_config.dist_url
    dist_utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    if dist_utils.is_main_process():
        print(f"\n{'='*60}")
        print(f"Thought Token Init Test")
        print(f"  num_thought_tokens: {args.num_thought_tokens}")
        print(f"  thought_init_token: '{args.thought_init_token}'")
        print(f"  lora_checkpoint: {args.lora_checkpoint}")
        print(f"  Output path_suffix: {config.experiment.path_suffix}")
        print(f"{'='*60}\n")
        print(OmegaConf.to_yaml(config, sort_keys=False))

    main(config, 
         lora_checkpoint=args.lora_checkpoint,
         thought_init_token=args.thought_init_token)

    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
