"""
Test script to verify Qwen3VL wrapper integration with MBEIR framework.
"""

import sys
import torch
import numpy as np
from PIL import Image

# Add paths
# 1. 把 Qwen3 的源码目录加进来，这样 wrapper 内部才能正常导入 Qwen3VLEmbedder
if '/data/Qwen3-VL-Embedding/src' not in sys.path:
    sys.path.insert(0, '/data/Qwen3-VL-Embedding/src')

# 2. 直接把 wrapper 所在的具体目录加进来，避开 models 冲突
wrapper_dir = '/data/LR1/src/models/qwen3'
if wrapper_dir not in sys.path:
    sys.path.insert(0, wrapper_dir)

# 3. 直接通过文件名导入，不再使用 models.qwen3 前缀
from qwen3_wrapper import Qwen3VLWrapper

def test_qwen3_wrapper():
    """Test Qwen3VL wrapper basic functionality."""
    print("=" * 80)
    print("Testing Qwen3VL Wrapper Integration")
    print("=" * 80)

    # Initialize wrapper
    # Initialize wrapper
    print("\n1. Initializing Qwen3VL wrapper...")
    try:
        wrapper = Qwen3VLWrapper(
            # ✅ 修改这里：确保这里传入的也是本地路径，或者直接删掉这行让它用上面的默认值
            model_name_or_path="/data/Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B",
            max_length=512,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("   ✓ Wrapper initialized successfully")
        print(f"   - Model device: {wrapper.device}")
        config = wrapper.embedder.model.config
        # 兼容性获取：先找顶层，找不到再去 text_config 里找
        embed_dim = getattr(config, "hidden_size", getattr(getattr(config, "text_config", None), "hidden_size", "Unknown"))
        print(f"   - Embedding dim: {embed_dim}")
    except Exception as e:
        print(f"   ✗ Failed to initialize wrapper: {e}")
        return False

    # Test text-only encoding
    print("\n2. Testing text-only encoding...")
    try:
        # Create mock batch
        batch_size = 2
        txt_batched = type('RawTextBatch', (), {
            '__getitem__': lambda self, idx: ["This is a test sentence.", "Another test sentence."] if isinstance(idx, list) else ["This is a test sentence.", "Another test sentence."][idx]
        })()

        image_batched = torch.zeros(batch_size, 3, 224, 224)
        txt_mask = torch.tensor([1, 1])
        image_mask = torch.tensor([0, 0])

        batch = {
            "txt_batched": txt_batched,
            "image_batched": image_batched,
            "txt_mask_batched": txt_mask,
            "image_mask_batched": image_mask,
            "did_list": ["doc1", "doc2"]
        }

        embeddings, ids = wrapper.encode_mbeir_batch(batch)
        print(f"   ✓ Text-only encoding successful")
        print(f"   - Embeddings shape: {embeddings.shape}")
        print(f"   - IDs: {ids}")
        print(f"   - Embedding norm: {torch.norm(embeddings[0]).item():.4f}")
    except Exception as e:
        print(f"   ✗ Text-only encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test image-only encoding
    print("\n3. Testing image-only encoding...")
    try:
        # Create random image
        image_batched = torch.rand(batch_size, 3, 224, 224)
        txt_mask = torch.tensor([0, 0])
        image_mask = torch.tensor([1, 1])

        batch = {
            "txt_batched": txt_batched,
            "image_batched": image_batched,
            "txt_mask_batched": txt_mask,
            "image_mask_batched": image_mask,
            "qid_list": ["query1", "query2"]
        }

        embeddings, ids = wrapper.encode_mbeir_batch(batch)
        print(f"   ✓ Image-only encoding successful")
        print(f"   - Embeddings shape: {embeddings.shape}")
        print(f"   - IDs: {ids}")
        print(f"   - Embedding norm: {torch.norm(embeddings[0]).item():.4f}")
    except Exception as e:
        print(f"   ✗ Image-only encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test multimodal encoding
    print("\n4. Testing multimodal encoding...")
    try:
        txt_mask = torch.tensor([1, 1])
        image_mask = torch.tensor([1, 1])

        batch = {
            "txt_batched": txt_batched,
            "image_batched": image_batched,
            "txt_mask_batched": txt_mask,
            "image_mask_batched": image_mask,
            "did_list": ["doc3", "doc4"]
        }

        embeddings, ids = wrapper.encode_mbeir_batch(batch)
        print(f"   ✓ Multimodal encoding successful")
        print(f"   - Embeddings shape: {embeddings.shape}")
        print(f"   - IDs: {ids}")
        print(f"   - Embedding norm: {torch.norm(embeddings[0]).item():.4f}")
    except Exception as e:
        print(f"   ✗ Multimodal encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_qwen3_wrapper()
    sys.exit(0 if success else 1)
