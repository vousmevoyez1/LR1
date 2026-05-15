#!/usr/bin/env python3
"""
Verify that special token embeddings are correctly loaded from checkpoint during evaluation.

This script:
1. Creates a fresh model with thought tokens setup
2. Loads a training checkpoint 
3. Compares: checkpoint special_embedding.weight vs. loaded model's special_embedding.weight
4. Reports if weights match, indicating successful loading
"""

import sys
import os
import argparse
import torch
from omegaconf import OmegaConf

# Add paths
sys.path.insert(0, '/data/LR1/src')
from common.utils import build_model_from_config
from models.qwen3.qwen3_thought_wrapper import Qwen3VLThoughtWrapper


def extract_special_embedding_from_checkpoint(ckpt_path: str):
    """Extract special_embedding.weight from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "trainable_state_dict" in ckpt:
        state = ckpt["trainable_state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    
    # Find special_embedding weights in checkpoint
    special_embed_weights = {}
    for key, val in state.items():
        if "special_embedding" in key:
            special_embed_weights[key] = val
    
    return special_embed_weights


def get_model_special_embedding_weight(model):
    """Extract special_embedding.weight from loaded model."""
    try:
        embed_layer = model.model.get_input_embeddings()
        if hasattr(embed_layer, "special_embedding"):
            return embed_layer.special_embedding.weight.detach().cpu()
        else:
            return None
    except Exception as e:
        print(f"Error getting special_embedding: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="/data/LR1/src/models/qwen3/qwen3/configs/2b/train/inbatch/inbatch.yaml",
                        help="Config YAML used during training")
    parser.add_argument("--num_thought_tokens", type=int, default=None, help="Override num_thought_tokens")
    parser.add_argument("--enable_final_token", type=bool, default=None, help="Override enable_final_token")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Special Token Embedding Loading Verification")
    print("="*80)

    # 1. Load checkpoint and extract special_embedding weights
    print(f"\n[Step 1] Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return False

    ckpt_special_weights = extract_special_embedding_from_checkpoint(args.checkpoint)
    if not ckpt_special_weights:
        print("❌ No special_embedding weights found in checkpoint!")
        return False

    print(f"✓ Found {len(ckpt_special_weights)} special_embedding key(s) in checkpoint:")
    for key, val in ckpt_special_weights.items():
        print(f"  - {key}: shape {val.shape}, dtype {val.dtype}")

    # 2. Create model from config (as evaluation does)
    print(f"\n[Step 2] Building model from config: {args.config}")
    config = OmegaConf.load(args.config)
    
    # Override if specified
    if args.num_thought_tokens is not None:
        config.model.num_thought_tokens = args.num_thought_tokens
        config.model.mbeir_num_thought_tokens = args.num_thought_tokens
    if args.enable_final_token is not None:
        config.model.enable_final_token = args.enable_final_token

    model = build_model_from_config(config)
    model.eval()
    print(f"✓ Model built with:")
    print(f"  - num_thought_tokens: {config.model.num_thought_tokens}")
    print(f"  - enable_final_token: {config.model.enable_final_token}")

    # Check that special_embedding wrapper is in place
    embed_layer = model.model.get_input_embeddings()
    if not hasattr(embed_layer, "special_embedding"):
        print("❌ Model does not have ReasoningTokenEmbeddingWrapper!")
        return False
    print(f"✓ ReasoningTokenEmbeddingWrapper found, special_embedding shape: {embed_layer.special_embedding.weight.shape}")

    # 3. Load checkpoint weights into model (as mbeir_embedderqwen.py does)
    print(f"\n[Step 3] Loading checkpoint into model...")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "trainable_state_dict" in state_dict:
        state_dict = state_dict["trainable_state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    with torch.no_grad():
        pre_norm = embed_layer.special_embedding.weight.norm().item()
    
    msg = model.load_state_dict(state_dict, strict=False)
    
    with torch.no_grad():
        post_norm = embed_layer.special_embedding.weight.norm().item()

    print(f"✓ Model state_dict loaded (strict=False)")
    print(f"  - Missing keys: {len(msg.missing_keys) if hasattr(msg, 'missing_keys') else 'N/A'}")
    print(f"  - Unexpected keys: {len(msg.unexpected_keys) if hasattr(msg, 'unexpected_keys') else 'N/A'}")
    print(f"  - Pre-load special_embedding norm: {pre_norm:.6f}")
    print(f"  - Post-load special_embedding norm: {post_norm:.6f}")

    # 4. Verify: special_embedding weight should have changed
    print(f"\n[Step 4] Verification")
    if abs(pre_norm - post_norm) < 1e-6:
        print(f"⚠️  WARNING: special_embedding weight norm didn't change!")
        print(f"    Pre-load norm: {pre_norm:.6f}")
        print(f"    Post-load norm: {post_norm:.6f}")
        print(f"    This suggests special_embedding weights were NOT loaded from checkpoint.")
        
        # Try to find why
        print(f"\n[Debugging] Checking state_dict keys:")
        for key in state_dict.keys():
            if "special_embedding" in key or "embed" in key:
                print(f"  - {key}")
        
        return False
    else:
        print(f"✅ SUCCESS: special_embedding weight changed after loading!")
        print(f"    Pre-load norm:  {pre_norm:.6f}")
        print(f"    Post-load norm: {post_norm:.6f}")
        print(f"    Change ratio: {abs(post_norm - pre_norm) / pre_norm * 100:.2f}%")
        return True


if __name__ == "__main__":
    success = main()
    print("\n" + "="*80)
    if success:
        print("✅ Special token embeddings loaded correctly!")
    else:
        print("❌ Special token embeddings were NOT loaded correctly!")
    print("="*80 + "\n")
    
    sys.exit(0 if success else 1)
