"""
Quick test to verify checkpoint loading is working correctly.

This script tests:
1. Model loading from base pretrained model
2. LoRA checkpoint loading
3. Basic forward pass with sample data

Run: python test_ckpt_loading.py
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, "/data/LR1/src")

from models.jina_v4.jina_v4.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model
from models.jina_v4.train import load_lora_checkpoint, freeze_base_model_keep_lora

# Checkpoint paths
CHECKPOINTS = {
    "MT0": "/data/LR1/checkpointMT0/jina_v4/Large/Instruct/InBatch/jina_v4_step_600.pth",
    "MT1": "/data/LR1/checkpointMT1/jina_v4/Large/Instruct/InBatch/jina_v4_step_100.pth",
    "MT3": "/data/LR1/checkpointMT3/jina_v4/Large/Instruct/InBatch/jina_v4_step_100.pth",
}

BASE_MODEL_PATH = "/data/jina-v4-local-copy"


def test_checkpoint_exists():
    """Test that all checkpoint files exist."""
    print("\n" + "="*60)
    print("Testing checkpoint file existence...")
    print("="*60)
    
    all_exist = True
    for name, path in CHECKPOINTS.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_checkpoint_structure(checkpoint_path: str):
    """Test checkpoint structure and keys."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print(f"  Keys: {list(ckpt.keys())}")
    
    if "lora_adapter" in ckpt:
        lora_keys = list(ckpt["lora_adapter"].keys())
        print(f"  LoRA adapter keys count: {len(lora_keys)}")
        if lora_keys:
            print(f"  Sample LoRA keys: {lora_keys[:3]}")
    
    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    
    if "global_step" in ckpt:
        print(f"  Global step: {ckpt['global_step']}")
    
    return ckpt


def test_model_loading(checkpoint_name: str, device: str = "cuda:0"):
    """Test loading model with checkpoint."""
    checkpoint_path = CHECKPOINTS[checkpoint_name]
    
    print("\n" + "="*60)
    print(f"Testing model loading for: {checkpoint_name}")
    print("="*60)
    
    # Load base model
    print("\n1. Loading base model...")
    model = JinaEmbeddingsV4Model.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print("   Base model loaded successfully!")
    
    # Set task
    model.task = 'retrieval'
    
    # Freeze base model
    print("\n2. Freezing base model, keeping LoRA...")
    model = freeze_base_model_keep_lora(model)
    
    # Count trainable parameters before loading checkpoint
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable_before:,}")
    
    # Load LoRA checkpoint
    print(f"\n3. Loading LoRA checkpoint: {checkpoint_path}")
    model, ckpt = load_lora_checkpoint(model, checkpoint_path)
    print("   LoRA checkpoint loaded successfully!")
    
    # Move to device
    print(f"\n4. Moving to device: {device}")
    model = model.to(device)
    model.eval()
    
    # Test encoding
    print("\n5. Testing text encoding...")
    model.mbeir_task_label = 'retrieval'
    model.mbeir_reason_steps = 0
    
    # Get base model for encoding
    if hasattr(model, 'base_model'):
        base_model = model.base_model.model
    else:
        base_model = model
    
    # Encode a test text
    test_text = "A photo of a cat sitting on a sofa"
    with torch.no_grad():
        embedding = base_model.encode_text(
            texts=[test_text],
            task='retrieval',
            return_numpy=False,
        )
    
    # encode_text returns a list when given a list, get first element
    if isinstance(embedding, list):
        embedding = embedding[0]
    
    print(f"   Input: '{test_text}'")
    print(f"   Output shape: {embedding.shape}")
    print(f"   Output dtype: {embedding.dtype}")
    print(f"   Output device: {embedding.device}")
    print(f"   Embedding norm: {torch.norm(embedding).item():.4f}")
    
    # Test with different reason_steps
    print("\n6. Testing with different reason_steps...")
    for rs in [0, 1, 3]:
        with torch.no_grad():
            emb = base_model.encode_text(
                texts=[test_text],
                task='retrieval',
                return_numpy=False,
                reason_steps=rs,
            )
        if isinstance(emb, list):
            emb = emb[0]
        print(f"   reason_steps={rs}: shape={emb.shape}, norm={torch.norm(emb).item():.4f}")
    
    print("\n✓ All tests passed!")
    
    return model


def main():
    print("\n" + "#"*70)
    print("# Checkpoint Loading Test")
    print("#"*70)
    
    # Test 1: Check file existence
    if not test_checkpoint_exists():
        print("\n✗ Some checkpoints are missing!")
        return
    
    # Test 2: Check checkpoint structure
    print("\n" + "="*60)
    print("Testing checkpoint structure...")
    print("="*60)
    
    for name, path in CHECKPOINTS.items():
        print(f"\n--- {name} ---")
        test_checkpoint_structure(path)
    
    # Test 3: Load model with each checkpoint
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Only test one checkpoint to save time/memory
    test_checkpoint = "MT0"
    model = test_model_loading(test_checkpoint, device)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "#"*70)
    print("# All tests completed successfully!")
    print("#"*70)


if __name__ == "__main__":
    main()
