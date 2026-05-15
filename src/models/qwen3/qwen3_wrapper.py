"""
Qwen3-VL Wrapper for MBEIR integration.

This module wraps Qwen3VLEmbedder to be compatible with the MBEIR framework.
Follows the same pattern as jina_v4's modeling_jina_embeddings_v4.py.

Reference usage from /data/Qwen3-VL-Embedding/test.py:
    model = Qwen3VLEmbedder(model_name_or_path=..., torch_dtype=..., attn_implementation=...)
    inputs = [{"text": "..."}, {"image": "path"}, {"text": "...", "image": PIL.Image}]
    embeddings = model.process(inputs)
"""

import sys
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from torchvision.transforms.functional import to_pil_image
from PIL import Image

# Add Qwen3-VL-Embedding to path
#sys.path.insert(0, '/data/Qwen3-VL-Embedding/src')
# from models.qwen3_vl_embedding import Qwen3VLEmbedder

# 明确告诉 Python 去 Qwen3 的源码目录里找
# 彻底避开 'models' 命名空间冲突
if '/data/Qwen3-VL-Embedding/src/models' not in sys.path:
    sys.path.insert(0, '/data/Qwen3-VL-Embedding/src/models')

from qwen3_vl_embedding import Qwen3VLEmbedder
class RawTextBatch:
    """Wrapper to store raw text strings for later processing.
    Must be defined at module level (not inside a function) for pickle compatibility
    with DataLoader's multiprocessing.
    """
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.texts[i] for i in idx]
        return self.texts[idx]


class Qwen3VLWrapper:
    """
    Wrapper class for Qwen3VLEmbedder to integrate with MBEIR framework.

    Provides:
    - encode_mbeir_batch(): core embedding interface
    - get_img_preprocess_fn(): image preprocessing for MBEIR data loader
    - get_tokenizer(): text passthrough for MBEIR data loader
    """

    def __init__(
        self,
        model_name_or_path: str = "/data/Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B",
        max_length: int = 1500,
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        **kwargs
    ):
        import os
        # 1. 动态获取当前进程被分配的 GPU 编号 (默认为 0，兼容单卡模式)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_str = f"cuda:{local_rank}"
        
        # 2. 强制把整个模型塞进这一张卡里，绝不跨卡
        # 技巧：在 Hugging Face 中，传入字典可以精确控制设备映射
        target_device_map = {"": device_str}

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_name_or_path,
            max_length=max_length,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=target_device_map, # ✅ 使用精确映射替换了 "auto"
            **kwargs
        )

        # MBEIR-specific attributes (set by build_model_from_config in common/utils.py)
        self.mbeir_task_label = 'retrieval'
        self.mbeir_image_size = (224, 224)
        self.mbeir_max_text_length = max_length
        self.task = 'retrieval'

    # ------------------------------------------------------------------
    # Required by mbeir_embedder.py (lines 553-560):
    #   img_preprocess_fn = model.get_img_preprocess_fn()
    #   tokenizer = model.get_tokenizer()
    # ------------------------------------------------------------------

    def get_img_preprocess_fn(self):
        """
        Returns image preprocessing function for MBEIR data loader.
        Only resizes and converts to tensor - NO normalization.
        This allows lossless conversion back to PIL in encode_mbeir_batch().
        """
        from torchvision import transforms

        def img_preprocess_wrapper(image: Image.Image) -> torch.Tensor:
            target_size = getattr(self, 'mbeir_image_size', (224, 224))
            if isinstance(target_size, int):
                target_size = (target_size, target_size)
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),  # [0, 1] range, no normalization
            ])
            return transform(image)

        return img_preprocess_wrapper

    def get_tokenizer(self):
        """
        Returns a passthrough function that preserves raw text strings.
        The MBEIR collator expects a callable returning something with len().
        We return RawTextBatch (same as jina_v4's pattern).
        """
        def passthrough_wrapper(texts: List[str]) -> RawTextBatch:
            return RawTextBatch(texts)

        return passthrough_wrapper

    # ------------------------------------------------------------------
    # Core embedding interface
    # ------------------------------------------------------------------

    def encode_mbeir_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode a batch from MBEIR dataset using Qwen3-VL's native processing.

        Handles three modality cases:
        - Text-only: txt_mask=1, image_mask=0
        - Image-only: txt_mask=0, image_mask=1
        - Multimodal: txt_mask=1, image_mask=1

        Args:
            batch: dict with txt_batched (RawTextBatch), image_batched ([B,3,H,W] in [0,1]),
                   txt_mask_batched, image_mask_batched, did_list/qid_list

        Returns:
            (embeddings [batch_size, embed_dim], id_list)
        """
        id_list = batch.get("did_list") or batch.get("qid_list")
        if id_list is None:
            raise ValueError("id_list (did_list or qid_list) not found in batch.")

        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask = batch["txt_mask_batched"]
        image_mask = batch["image_mask_batched"]

        batch_size = image_batched.size(0)
        device = self.embedder.model.device

        # Group indices by modality
        text_only_idx, image_only_idx, multimodal_idx = [], [], []
        for i in range(batch_size):
            has_text = txt_mask[i].item() == 1
            has_image = image_mask[i].item() == 1
            if has_text and has_image:
                multimodal_idx.append(i)
            elif has_image:
                image_only_idx.append(i)
            elif has_text:
                text_only_idx.append(i)

        # 动态获取向量维度
        config = self.embedder.model.config
        if hasattr(config, "hidden_size"):
            embed_dim = config.hidden_size
        elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            embed_dim = config.text_config.hidden_size
        else:
            embed_dim = 1536 # Qwen3-VL-2B 的默认 fallback 维度

        # 顺便修复数据类型匹配问题，防止 bf16 与 fp32 冲突
        embeddings = torch.zeros(batch_size, embed_dim, device=device, dtype=self.embedder.model.dtype)

        # Process each modality group via Qwen3VLEmbedder.process()
        # Following pattern from /data/Qwen3-VL-Embedding/test.py:
        #   inputs = [{"text": "..."}, {"image": PIL}, {"text": "...", "image": PIL}]
        #   embeddings = model.process(inputs)
        if text_only_idx:
            embeddings[text_only_idx] = self._encode_text_batch(txt_batched, text_only_idx).to(embeddings.dtype)
        if image_only_idx:
            embeddings[image_only_idx] = self._encode_image_batch(image_batched, image_only_idx).to(embeddings.dtype)
        if multimodal_idx:
            embeddings[multimodal_idx] = self._encode_multimodal_batch(txt_batched, image_batched, multimodal_idx).to(embeddings.dtype)

        return embeddings, id_list

    def _encode_text_batch(self, txt_batched, indices: List[int]) -> torch.Tensor:
        """Encode text-only samples. Pattern: {"text": "..."} from test.py"""
        texts = txt_batched[indices]
        inputs = [{'text': text} for text in texts]
        return self.embedder.process(inputs, normalize=True)

    def _encode_image_batch(self, image_batched: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """Encode image-only samples. Pattern: {"image": PIL.Image} from test.py"""
        pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in indices]
        inputs = [{'image': img} for img in pil_images]
        return self.embedder.process(inputs, normalize=True)

    def _encode_multimodal_batch(self, txt_batched, image_batched: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """Encode multimodal samples. Pattern: {"text": "...", "image": PIL.Image} from test.py"""
        texts = txt_batched[indices]
        pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in indices]
        inputs = [{'text': text, 'image': img} for text, img in zip(texts, pil_images)]
        return self.embedder.process(inputs, normalize=True)

    # ------------------------------------------------------------------
    # Model lifecycle methods (required by mbeir_embedder.py)
    # ------------------------------------------------------------------

    def forward(self, encode_mbeir_batch: bool = False, **kwargs):
        """Forward pass. mbeir_embedder.py calls: model(encode_mbeir_batch=True, **batch)"""
        if encode_mbeir_batch:
            return self.encode_mbeir_batch(kwargs)
        else:
            raise NotImplementedError("Only encode_mbeir_batch mode is supported")

    def __call__(self, encode_mbeir_batch: bool = False, **kwargs):
        return self.forward(encode_mbeir_batch=encode_mbeir_batch, **kwargs)

    def eval(self):
        self.embedder.model.eval()
        return self

    def train(self, mode: bool = True):
        self.embedder.model.train(mode)
        return self

    def to(self, device):
        self.embedder.model = self.embedder.model.to(device)
        return self

    def cuda(self, device=None):
        self.embedder.model = self.embedder.model.cuda(device)
        return self

    def parameters(self):
        return self.embedder.model.parameters()

    def named_parameters(self):
        return self.embedder.model.named_parameters()

    @property
    def device(self):
        return self.embedder.model.device
