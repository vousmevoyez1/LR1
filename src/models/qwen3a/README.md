# Qwen3-VL Integration for MBEIR Benchmark

This directory contains the integration of Qwen3-VL-Embedding model with the MBEIR (Multimodal BEIR) benchmark framework.

## Directory Structure

```
qwen3/
├── __init__.py                 # Package initialization
├── engine.py                   # Training/evaluation engine (compatible with jina_v4)
├── utils.py                    # Utility functions
├── qwen3_wrapper.py           # Wrapper class for MBEIR integration
├── test_qwen3_wrapper.py      # Test script for wrapper
└── qwen3/
    └── configs/
        └── 2b/
            └── eval/
                └── inbatch/
                    ├── embedood.yaml       # Embedding config
                    ├── indexood.yaml       # Indexing config
                    ├── retrievalood.yaml   # Retrieval config
                    └── eval_qwen3_vl.sh    # Evaluation script
```

## Model Information

- **Model**: Qwen3-VL-Embedding-2B
- **Embedding Dimension**: 2048
- **Max Sequence Length**: 8192 tokens
- **Supported Modalities**: Text, Image, Multimodal (Text+Image)

## Setup

1. Ensure Qwen3-VL-Embedding repository is available at `/data/Qwen3-VL-Embedding/`
2. Install required dependencies:
   ```bash
   pip install transformers torch torchvision pillow qwen-vl-utils
   ```

## Usage

### Testing the Integration

Run the test script to verify the wrapper works correctly:

```bash
cd /data/LR1/src/models/qwen3
python test_qwen3_wrapper.py
```

### Running MBEIR Evaluation

1. Navigate to the config directory:
   ```bash
   cd /data/LR1/src/models/qwen3/qwen3/configs/2b/eval/inbatch
   ```

2. Run the evaluation script:
   ```bash
   bash eval_qwen3_vl.sh
   ```

This will:
- Generate embeddings for queries and candidate pools
- Create FAISS indices
- Perform retrieval and compute metrics

### Configuration

Edit the YAML files to customize:
- `embedood.yaml`: Embedding generation settings
  - `batch_size`: Adjust based on GPU memory
  - `mbeir_max_text_length`: Maximum text length (default: 8192)
  - `datasets_name`: Datasets to evaluate

- `indexood.yaml`: Index creation settings
  - `dim`: Embedding dimension (2048 for 2B model)
  - `idx_type`: FAISS index type (Flat, IVF, etc.)

- `retrievalood.yaml`: Retrieval settings
  - `correspond_metrics_name`: Metrics to compute (Recall@K)

## Architecture

### Qwen3VLWrapper

The `Qwen3VLWrapper` class wraps `Qwen3VLEmbedder` to be compatible with MBEIR:

```python
from models.qwen3.qwen3_wrapper import Qwen3VLWrapper

# Initialize wrapper
wrapper = Qwen3VLWrapper(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
    max_length=8192,
    torch_dtype=torch.bfloat16,
)

# Encode MBEIR batch
embeddings, ids = wrapper.encode_mbeir_batch(batch)
```

### Key Features

1. **Modality Handling**: Automatically detects and processes text-only, image-only, and multimodal inputs
2. **Batch Processing**: Efficient batch encoding with proper grouping by modality
3. **MBEIR Compatible**: Implements `encode_mbeir_batch` interface expected by MBEIR framework
4. **Normalized Embeddings**: Returns L2-normalized embeddings for cosine similarity

## Benchmark Datasets

The evaluation covers out-of-distribution (OOD) datasets:
- **edis_task2**: Entity disambiguation with images
- **nights_task4**: Natural image-text retrieval
- **fashioniq_task7**: Fashion image retrieval with text queries
- **cirr_task7**: Composed image retrieval

## Comparison with Jina-V4

| Feature | Jina-V4 | Qwen3-VL |
|---------|---------|----------|
| Embedding Dim | 2048 | 2048 (2B) / 4096 (8B) |
| Max Length | 512 | 8192 |
| Reasoning Steps | Supported | Not applicable |
| LoRA Adapters | Supported | Not applicable |
| Native Multimodal | Yes | Yes |

## Notes

- Qwen3-VL uses a different architecture than Jina-V4, so reasoning steps and LoRA adapters are not applicable
- The model uses last token pooling instead of mean pooling
- Batch size may need to be reduced compared to Jina-V4 due to longer sequence length support

## Troubleshooting

1. **CUDA Out of Memory**: Reduce `batch_size` in `embedood.yaml`
2. **Model Download Issues**: Ensure Hugging Face access token is configured if needed
3. **Import Errors**: Verify `/data/Qwen3-VL-Embedding/src` is accessible

## References

- [Qwen3-VL-Embedding Repository](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [MBEIR Benchmark](https://github.com/TIGER-AI-Lab/UniIR)
