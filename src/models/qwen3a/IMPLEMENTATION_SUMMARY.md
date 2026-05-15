# Qwen3 Thought Token Implementation - Summary

## Overview
Successfully ported the Jina-v4oa thought token mechanism to Qwen3-VL, enabling:
- Reasoning tokens (`<thought_1..K>` + `<final>`) for query representation
- Attention mask constraints where `<final>` only attends to thought tokens
- Asymmetric query/doc encoding (queries use thought tokens, documents don't)
- Training support for LoRA + new token embeddings only

## Files Modified/Created

### 1. `/data/LR1/src/models/qwen3/qwen3_thought_wrapper.py` (NEW)
**Purpose**: Main wrapper implementing thought token mechanism for Qwen3-VL

**Key Components**:
- `Qwen3VLThoughtWrapper`: Replaces the old `Qwen3VLWrapper`
- `setup_thought_tokens()`: Adds tokens to tokenizer, resizes embeddings, initializes randomly
- `_build_reasoning_attention_mask()`: Creates 4D mask restricting `<final>` attention
- `_forward_embeddings()`: Trainable forward pass with mask injection
- `_get_single_vector_embedding()`: Extracts `<final>` token embedding for queries
- `encode_mbeir_batch()`: Asymmetric encoding (queries use thought tokens, docs don't)

### 2. `/data/LR1/src/models/qwen3/engine.py` (MODIFIED)
**Changes**:
- Added `num_thought_tokens` parameter to `encode_batch_for_training()`
- Updated to call `model._preprocess_inputs()` and `model._forward_embeddings()`
- Modified `train_one_epoch()`: queries use `num_thought_tokens=K`, candidates use `0`
- Modified `eval_engine()`: same asymmetric encoding for evaluation

### 3. `/data/LR1/src/models/qwen3/utils.py` (MODIFIED)
**Changes**:
- Added `freeze_base_model_keep_lora()` function
- Freezes base model, keeps LoRA trainable
- Uses gradient hook to allow only thought/final token embedding updates

### 4. `/data/LR1/src/common/utils.py` (MODIFIED)
**Changes**:
- Updated `build_model_from_config()` for Qwen3VLEmbedder
- Now imports `Qwen3VLThoughtWrapper` instead of old wrapper
- Calls `setup_thought_tokens()` if `mbeir_num_thought_tokens > 0`
- Supports config parameters: `num_thought_tokens`, `semantic_init_token`, `skip_thought_init`

### 5. `/data/LR1/src/models/qwen3/test_thought_tokens.py` (NEW)
**Purpose**: Sanity check script with 4 tests:
1. Thought token setup and ID uniqueness
2. Attention mask construction and constraints
3. Forward pass with thought tokens
4. Query/doc asymmetric encoding

## Key Design Decisions

### 1. Trainable Forward vs `.process()`
- **Old approach**: Used `Qwen3VLEmbedder.process()` (no_grad wrapper)
- **New approach**: Direct forward through `Qwen3VLForEmbedding` model
- **Reason**: Enables gradient flow for training LoRA + token embeddings

### 2. Attention Mask Format
- Uses 4D mask `[batch, 1, seq_len, seq_len]` with float values (0=allow, -inf=mask)
- Only modifies the `<final>` token's attention row
- Preserves standard causal attention for all other tokens

### 3. Token ID Uniqueness
- Validates `len(set(all_ids)) == len(all_ids)` after adding tokens
- Raises error if duplicate IDs detected
- Critical for correct attention mask construction

### 4. Asymmetric Encoding
- Determined by batch type: `qid_list` (query) vs `did_list` (document)
- Queries: `num_thought_tokens = config.mbeir_num_thought_tokens`
- Documents: `num_thought_tokens = 0`
- Implemented in both `encode_mbeir_batch()` and training engine

## Configuration Example

```yaml
model:
  model_name: "Qwen3VLEmbedder"
  original_model_name: "/data/Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B"
  mbeir_task_label: "retrieval"
  mbeir_max_text_length: 8192
  mbeir_num_thought_tokens: 8  # Number of thought tokens (K)
  semantic_init_token: "."      # Optional: token for semantic init
  skip_thought_init: false      # Set true when loading checkpoint
  num_thought_tokens: 8         # For training engine
  temperature: 0.07
```

## Training Setup

```python
from models.qwen3.qwen3_thought_wrapper import Qwen3VLThoughtWrapper
from models.qwen3.utils import freeze_base_model_keep_lora

# 1. Load model
model = Qwen3VLThoughtWrapper(...)

# 2. Setup thought tokens
model.setup_thought_tokens(num_thought_tokens=8)

# 3. Freeze base model, keep LoRA + thought embeddings trainable
thought_token_ids = model._thought_token_ids + [model._final_token_id]
freeze_base_model_keep_lora(model, thought_token_ids=thought_token_ids)

# 4. Train with InfoNCE loss
# - Queries encoded with num_thought_tokens=8
# - Candidates encoded with num_thought_tokens=0
# - Loss = InfoNCE(e_final, Doc+)
```

## Verification Steps

### Run Sanity Checks
```bash
cd /data/LR1/src
python models/qwen3/test_thought_tokens.py
```

Expected output:
```
✓ Thought token IDs: [151659, 151660, 151661, 151662]
✓ Final token ID: 151663
✓ All IDs are unique
✓ Attention mask shape: (2, 1, 20, 20)
✓ Final token correctly restricted to thought tokens + itself
✓ Forward pass successful
✓ Output shape: (1, 1536)
✓ Embedding is normalized
✓ Query encoding successful: (2, 1536)
✓ Doc encoding successful: (2, 1536)
✓ ALL TESTS PASSED
```

### Validate Integration
```bash
python /data/LR1/src/models/qwen3/validate_setup.py
```

## Differences from Jina-v4oa

| Aspect | Jina-v4oa | Qwen3 |
|--------|-----------|-------|
| Base Model | Qwen2.5-VL (via PEFT) | Qwen3-VL (direct) |
| LoRA | Built-in PEFT support | Manual freeze + hook |
| Processor | Custom JinaEmbeddingsV4Processor | Qwen3VLProcessor |
| Image Processing | Qwen2.5-VL vision encoder | Qwen3-VL vision encoder |
| Multi-vector | Supported | Not implemented |
| Thought Analysis | `encode_at_step()` supported | Not implemented |

## Next Steps

1. **Training**: Create training config with `num_thought_tokens` parameter
2. **Checkpoint**: Implement save/load for thought token embeddings
3. **Evaluation**: Run MBEIR evaluation with thought tokens enabled
4. **Ablation**: Compare performance with/without thought tokens

## Notes

- The implementation follows the exact pattern from Jina-v4oa's `modeling_jina_embeddings_v4.py`
- All core constraints are satisfied:
  - ✓ Thought tokens have unique IDs
  - ✓ `<final>` only attends to thought tokens
  - ✓ Query uses `<final>` embedding
  - ✓ Asymmetric query/doc encoding
  - ✓ InfoNCE loss with e_final
- Compatible with existing MBEIR training/evaluation infrastructure
