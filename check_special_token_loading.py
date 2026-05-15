#!/usr/bin/env python3
"""
检查 special token embedding 是否被正确加载的诊断脚本。

这脚本验证：
1. ReasoningTokenEmbeddingWrapper 是否被正确安装
2. special_embedding 权重是否被正确初始化和加载
3. train.py 保存的特殊 token 权重是否在 eval 时被恢复
"""

import sys
import os
import torch
from omegaconf import OmegaConf

# Setup paths
sys.path.insert(0, '/data/LR1/src')
sys.path.insert(0, '/data/Qwen3-VL-Embedding/src')

from common.utils import build_model_from_config
from models.qwen3.train import _collect_trainable_state_dict
from models.qwen3.qwen3_thought_wrapper import ReasoningTokenEmbeddingWrapper


def check_model_setup(config_path: str, checkpoint_path: str = None):
    """
    检查模型初始化和checkpoint加载逻辑。
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 可选的 checkpoint 路径
    """
    print("\n" + "="*80)
    print("Special Token Embedding Loading Check")
    print("="*80)
    
    # Load config
    print("\n[1] Loading config...")
    config = OmegaConf.load(config_path)
    print(f"✓ Config loaded: {config_path}")
    
    # Build model (with thought tokens)
    print("\n[2] Building model with thought token setup...")
    model = build_model_from_config(config)
    model.eval()
    print(f"✓ Model built")
    
    # Check if ReasoningTokenEmbeddingWrapper was installed
    print("\n[3] Checking embedding layer wrapping...")
    embed_layer = model.model.get_input_embeddings()
    is_wrapped = isinstance(embed_layer, ReasoningTokenEmbeddingWrapper)
    print(f"   Embedding layer type: {type(embed_layer).__name__}")
    print(f"   Is ReasoningTokenEmbeddingWrapper: {is_wrapped}")
    
    if is_wrapped:
        print(f"   Special token IDs: {embed_layer.special_token_ids}")
        print(f"   Special embedding shape: {embed_layer.special_embedding.weight.shape}")
        initial_norm = embed_layer.special_embedding.weight.norm().item()
        print(f"   ✓ Initial special embedding norm: {initial_norm:.6f}")
    else:
        print("   ✗ WARNING: Embedding layer is NOT wrapped! Thought tokens may not be trainable.")
        return
    
    # Get model state before checkpoint loading
    if is_wrapped:
        embed_before = embed_layer.special_embedding.weight.clone().detach()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n[4] Loading checkpoint: {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if "trainable_state_dict" in checkpoint:
            state_dict = checkpoint["trainable_state_dict"]
            print(f"✓ Found 'trainable_state_dict' with {len(state_dict)} entries")
            
            # Check for special_embedding keys
            special_embed_keys = [k for k in state_dict.keys() if "special_embedding" in k]
            print(f"   Special embedding keys in checkpoint: {len(special_embed_keys)}")
            if special_embed_keys:
                for key in special_embed_keys:
                    print(f"     - {key}: shape={state_dict[key].shape}, dtype={state_dict[key].dtype}")
                    print(f"       norm before: {embed_layer.special_embedding.weight.norm():.6f}")
            else:
                print("   ✗ WARNING: No 'special_embedding' keys found in trainable_state_dict!")
            
            # Load state
            print("\n[5] Loading state dict into model...")
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"✓ Load message: {msg}")
            
            # Check if special_embedding was updated
            if is_wrapped:
                embed_after = embed_layer.special_embedding.weight.clone().detach()
                
                # Compare norms
                embed_before_norm = embed_before.norm().item()
                embed_after_norm = embed_after.norm().item()
                
                print(f"\n[6] Special embedding update check:")
                print(f"   Norm before loading: {embed_before_norm:.6f}")
                print(f"   Norm after loading:  {embed_after_norm:.6f}")
                print(f"   Difference: {abs(embed_before_norm - embed_after_norm):.6f}")
                
                # Check for actual changes
                weight_diff = (embed_before - embed_after).abs().max().item()
                print(f"   Max weight difference: {weight_diff:.6f}")
                
                if weight_diff < 1e-6:
                    print("   ✗ CRITICAL: Special embedding weights did NOT change after loading!")
                    print("   This means the checkpoint weights were NOT applied.")
                else:
                    print(f"   ✓ Special embedding weights were updated successfully!")
        else:
            print(f"✗ Checkpoint does NOT have 'trainable_state_dict'")
            print(f"   Available keys: {checkpoint.keys()}")
    else:
        if checkpoint_path:
            print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        else:
            print("\n[4] Skipped checkpoint loading (no path provided)")
    
    print("\n[7] Summary of state dict keys related to special tokens:")
    wrapper_state = model.state_dict()
    special_keys = [k for k in wrapper_state.keys() if "special_embedding" in k]
    if special_keys:
        for key in special_keys:
            print(f"   ✓ {key}: shape={wrapper_state[key].shape}")
    else:
        print("   ✗ No 'special_embedding' keys in model state dict!")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check special token embedding loading")
    parser.add_argument(
        "--config",
        type=str,
        default="/data/LR1/src/models/qwen3/qwen3/configs/2b/eval/inbatch/embed.yaml",
        help="Path to embed config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to checkpoint to verify loading",
    )
    args = parser.parse_args()
    
    check_model_setup(args.config, args.checkpoint)
