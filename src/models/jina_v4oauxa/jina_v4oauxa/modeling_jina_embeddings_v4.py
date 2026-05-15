# Jina Embeddings V4 Model implementation was inspired by the ColPali codebase:
# https://github.com/illuin-tech/colpali

import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from io import BytesIO
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import requests
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchFeature
from transformers.utils import is_flash_attn_2_available

from .configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
from .custom_lora_module import MultiAdapterLinear
from .qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


class PromptType(str, Enum):
    query = "query"
    passage = "passage"


PREFIX_DICT = {"query": "Query", "passage": "Passage"}

###
class RawTextBatch:
    """Wrapper to store raw text strings for later processing."""
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.texts[i] for i in idx]
        return self.texts[idx]
###


# ==================== Reasoning Token Constants ====================
# These special tokens are used for implicit multi-step reasoning
THOUGHT_TOKEN_PREFIX = "<thought_"
THOUGHT_TOKEN_SUFFIX = ">"
MAX_THOUGHT_TOKENS = 16  # Maximum number of thought tokens supported


def get_thought_token(idx: int) -> str:
    """Get the thought token string for a given index (1-indexed)."""
    return f"{THOUGHT_TOKEN_PREFIX}{idx}{THOUGHT_TOKEN_SUFFIX}"


def get_all_thought_tokens(num_tokens: int) -> List[str]:
    """Get a list of all thought tokens up to num_tokens."""
    return [get_thought_token(i) for i in range(1, num_tokens + 1)]


class JinaEmbeddingsV4Processor(Qwen2_5_VLProcessor):
    def __init__(self, *args, **kwargs) -> None:
        Qwen2_5_VLProcessor.__init__(self, *args, **kwargs)
        self.assistant_prefix_len = 58
        self.text_max_length = 32768
        self._thought_tokens_added = False
        self._num_thought_tokens = 0
        
    def add_thought_tokens(self, num_tokens: int = 8) -> List[int]:
        """
        Add thought tokens to the tokenizer vocabulary.
        
        Args:
            num_tokens: Number of thought tokens to add (e.g., 8 means <thought_1> to <thought_8>)
            
        Returns:
            List of token IDs for the added thought tokens
        """
        if self._thought_tokens_added and self._num_thought_tokens >= num_tokens:
            # Already added enough tokens
            return [self.tokenizer.convert_tokens_to_ids(get_thought_token(i)) 
                    for i in range(1, num_tokens + 1)]
        
        thought_tokens = get_all_thought_tokens(num_tokens)
        
        # Add special tokens to tokenizer
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': thought_tokens
        })
        
        if num_added > 0:
            print(f"Added {num_added} thought tokens to tokenizer: {thought_tokens}")
        
        self._thought_tokens_added = True
        self._num_thought_tokens = num_tokens
        
        # Return the token IDs
        return [self.tokenizer.convert_tokens_to_ids(t) for t in thought_tokens]
    
    def get_thought_token_ids(self, num_tokens: int) -> List[int]:
        """Get the token IDs for the specified number of thought tokens."""
        if not self._thought_tokens_added:
            return self.add_thought_tokens(num_tokens)
        return [self.tokenizer.convert_tokens_to_ids(get_thought_token(i)) 
                for i in range(1, num_tokens + 1)]

    def _append_thought_tokens(self, text: str, num_thought_tokens: int) -> str:
        """Append thought tokens to a text string."""
        if num_thought_tokens <= 0:
            return text
        thought_suffix = " " + " ".join(get_all_thought_tokens(num_thought_tokens))
        return text + thought_suffix

    def process_images(
        self,
        images: Union[List[Image.Image], List[List[Image.Image]]],
        num_thought_tokens: int = 0,
    ) -> BatchFeature:
        """
        Process images with optional thought tokens.
        
        Args:
            images: List of PIL Images
            num_thought_tokens: Number of thought tokens to append (0 = disabled)
        """
        # Ensure thought tokens are added to vocabulary if needed
        if num_thought_tokens > 0:
            self.add_thought_tokens(num_thought_tokens)

        if isinstance(images[0], list):
            images = cast(List[List[Image.Image]], images)
            text_doc = []
            for i in range(len(images)):
                conversation = [
                    {"role": "user", "content": [{"type": "image"}] * len(images[i])}
                ]
                template = self.apply_chat_template(
                    conversation, add_generation_prompt=False
                )
                text = template[self.assistant_prefix_len :]
                # Append thought tokens if specified
                if num_thought_tokens > 0:
                    text = self._append_thought_tokens(text, num_thought_tokens)
                text_doc.append(text)
        else:
            images = cast(List[Image.Image], images)
            base_text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
            if num_thought_tokens > 0:
                base_text = self._append_thought_tokens(base_text, num_thought_tokens)
            text_doc = [base_text] * len(images)

        # The following code is a hack to make sure the scatter in DDP is done correctly when training on multiple GPUs
        batch_doc = self(text=text_doc, images=images, padding="longest", return_tensors="pt")
        # Separate pixel_values for each image
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]
        # Pad pixel_values to the same length to be able to make it into a tensor
        pixel_values = torch.split(batch_doc["pixel_values"], offsets.tolist())

        max_length = max([len(pv) for pv in pixel_values])

        pixel_values = [
            torch.cat(
                [
                    pv,
                    torch.zeros(
                        (max_length - len(pv), pv.shape[1]),
                        dtype=pv.dtype,
                        device=pv.device,
                    ),
                ]
            )
            for pv in pixel_values
        ]

        batch_doc["pixel_values"] = torch.stack(pixel_values)
        return batch_doc

    def process_texts(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        prefix: Optional[str] = None,
        padding: Optional[str] = None,
        num_thought_tokens: int = 0,
    ) -> BatchFeature:
        """
        Process texts with optional thought tokens.
        
        Args:
            texts: List of text strings
            max_length: Maximum token length
            prefix: Optional prefix for text
            padding: Padding strategy
            num_thought_tokens: Number of thought tokens to append (0 = disabled)
        """
        # Ensure thought tokens are added to vocabulary if needed
        if num_thought_tokens > 0:
            self.add_thought_tokens(num_thought_tokens)

        max_length = (
            self.text_max_length
            if max_length is None
            else min(max_length, self.text_max_length)
        )
        padded_texts: List[str] = []

        for text in texts:
            if prefix:
                assert 1==2, "process_texts of Jina-v4 does not support prefix anymore."
                text = f"{prefix}: {text}"
            # Append thought tokens if specified
            if num_thought_tokens > 0:
                text = self._append_thought_tokens(text, num_thought_tokens)
            padded_texts.append(text)

        text_batch = self(
            text=padded_texts,
            return_tensors="pt",
            padding=padding or "longest",
            max_length=max_length,
            truncation=True,
        )

        return text_batch
    
    def process_multimodal(
        self,
        images: List[Image.Image],
        texts: List[str],
        max_length: Optional[int] = None,
        prefix: Optional[str] = None,
        num_thought_tokens: int = 0,
    ) -> BatchFeature:
        """
        Process multimodal (image + text) inputs with optional thought tokens.
        
        Creates prompts in the format:
        <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>\n [<thought_1>...<thought_K>]

        Args:
            images: List of PIL Images
            texts: List of text strings (one per image)
            max_length: Maximum token length
            prefix: Optional prefix for text (e.g., "Query")
            num_thought_tokens: Number of thought tokens to append (0 = disabled)

        Returns:
            BatchFeature with input_ids, attention_mask, pixel_values, image_grid_thw
        """
        # Ensure thought tokens are added to vocabulary if needed
        if num_thought_tokens > 0:
            self.add_thought_tokens(num_thought_tokens)
            
        max_length = (
            self.text_max_length
            if max_length is None
            else min(max_length, self.text_max_length)
        )
        overhead_tokens = len(self.tokenizer.encode(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n", 
            add_special_tokens=False
        ))
        # Account for thought tokens in max length calculation
        thought_tokens_len = num_thought_tokens + 1 if num_thought_tokens > 0 else 0  # +1 for space
        text_max_length = max_length - overhead_tokens - 256 - thought_tokens_len
    
        # Construct multimodal prompts
        text_prompts = []
        for text in texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) > text_max_length:
                encoded = encoded[:text_max_length]
                text = self.tokenizer.decode(encoded)
            prompt = f"<|im_start|>user\n{text}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
            # Append thought tokens if specified
            if num_thought_tokens > 0:
                prompt = self._append_thought_tokens(prompt, num_thought_tokens)
            text_prompts.append(prompt)

        # Process with both text and images
        batch = self(
            text=text_prompts,
            images=images,
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Handle pixel_values padding (same as process_images)
        offsets = batch["image_grid_thw"][:, 1] * batch["image_grid_thw"][:, 2]
        pixel_values = torch.split(batch["pixel_values"], offsets.tolist())
        max_pv_length = max([len(pv) for pv in pixel_values])

        pixel_values = [
            torch.cat([
                pv,
                torch.zeros(
                    (max_pv_length - len(pv), pv.shape[1]),
                    dtype=pv.dtype,
                    device=pv.device,
                ),
            ])
            for pv in pixel_values
        ]
        batch["pixel_values"] = torch.stack(pixel_values)

        return batch


@dataclass
class JinaEmbeddingsV4ModelOutput:
    """
    Base class for the Hybrid Model outputs.
    Args:
        vlm_last_hidden_states (torch.Tensor, optional): Last hidden states of the VLM.
        single_vec_emb (torch.Tensor, optional): Single-vector embeddings (from last thought token).
        multi_vec_emb (torch.Tensor, optional): Multi-vector embeddings.
        thought_embeddings (List[torch.Tensor], optional): Embeddings from each thought token position.
            Used for deep supervision during training.
    """

    vlm_last_hidden_states: Optional[torch.Tensor] = None
    single_vec_emb: Optional[torch.Tensor] = None
    multi_vec_emb: Optional[torch.Tensor] = None
    thought_embeddings: Optional[List[torch.Tensor]] = None
    hard_gate_logits: Optional[torch.Tensor] = None
    selected_thought_indices: Optional[torch.Tensor] = None


class StepPredictorGate(nn.Module):
    """
    Step predictor gate.

    输入所有 thought token embedding（[B, K, D]）以及可选 query 上下文 embedding（[B, D]），
    对每个 thought step 进行打分并输出 logits（[B, K]）。
    """

    def __init__(self, hidden_dim: int, num_steps: int, mlp_hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_steps = int(num_steps)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        thought_states: torch.Tensor,
        query_context_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            thought_states: [B, K, D]，所有 thought token embedding
            query_context_embeddings: [B, D]，query 编码后的上下文向量。
                若为 None，则使用 thought_states 的均值向量作为 context。

        Returns:
            step_logits: [B, K]
        """
        if thought_states.dim() != 3:
            raise ValueError(
                f"Expected thought_states with shape [B, K, D], got {tuple(thought_states.shape)}"
            )

        batch_size, num_steps, hidden_dim = thought_states.shape
        if num_steps != self.num_steps:
            raise ValueError(
                f"StepPredictorGate expected K={self.num_steps}, got K={num_steps}."
            )

        if query_context_embeddings is None:
            context = thought_states.mean(dim=1)
        else:
            if query_context_embeddings.dim() != 2:
                raise ValueError(
                    f"Expected query_context_embeddings with shape [B, D], "
                    f"got {tuple(query_context_embeddings.shape)}"
                )
            if query_context_embeddings.shape != (batch_size, hidden_dim):
                raise ValueError(
                    f"query_context_embeddings shape mismatch: expected {(batch_size, hidden_dim)}, "
                    f"got {tuple(query_context_embeddings.shape)}"
                )
            context = query_context_embeddings

        context_expand = context.unsqueeze(1).expand(-1, num_steps, -1)
        gate_inputs = torch.cat([thought_states, context_expand], dim=-1)
        step_logits = self.net(gate_inputs).squeeze(-1)
        return step_logits


class JinaEmbeddingsV4Model(Qwen2_5_VLForConditionalGeneration):
    config_class = JinaEmbeddingsV4Config
    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(self, config: JinaEmbeddingsV4Config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        self._init_projection_layer(config)
        self.post_init()
        self.processor = JinaEmbeddingsV4Processor.from_pretrained(
            self.name_or_path, trust_remote_code=True, use_fast=True
        )
        self.multi_vector_projector_dim = config.multi_vector_projector_dim
        self.verbosity = config.verbosity
        self._task = None
        
        # Thought token configuration
        self._num_thought_tokens = 0
        self._thought_token_ids = None
        # MBEIR encoding mode: False=asymmetric (query-only thought tokens), True=symmetric (query+candidate)
        self.mbeir_symmetric_encoding = False
        # Final embedding mode when thought tokens are used:
        # - "last": use <thought_K>
        # - "mean_all_thought_tokens": mean-pool all <thought_i>
        self.thought_pooling_mode = "last"
        self.mbeir_thought_pooling_mode = "last"

        # Step predictor gate (stage-2 routing module)
        self.step_predictor_gate: Optional[StepPredictorGate] = None
        # Backward compatibility alias for existing training code/checkpoints.
        self.hard_gate = None
        self._hard_gate_selection_history: List[List[int]] = []

        # Debug switches for inspecting exact encoded model inputs.
        # Enable via:
        #   export JINA_V4_DEBUG_INPUT=1
        # Optional:
        #   export JINA_V4_DEBUG_INPUT_MAX_ROWS=8
        self.debug_input_dump = bool(int(os.environ.get("JINA_V4_DEBUG_INPUT", "0")))
        self.debug_input_max_rows = int(os.environ.get("JINA_V4_DEBUG_INPUT_MAX_ROWS", "4"))

    def setup_step_predictor_gate(
        self,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        force_reinit: bool = False,
        num_steps: Optional[int] = None,
    ) -> None:
        """Initialize (or keep) the StepPredictorGate routing module."""
        hidden_dim = int(self.config.text_config.hidden_size)
        if mlp_hidden_dim is None:
            mlp_hidden_dim = max(hidden_dim // 4, 128)
        if num_steps is None:
            num_steps = int(self._num_thought_tokens)

        if num_steps <= 0:
            raise ValueError(
                "num_steps must be > 0 for StepPredictorGate. "
                "Call setup_thought_tokens(K) before setup_step_predictor_gate()."
            )

        if self.step_predictor_gate is not None and not force_reinit:
            return

        self.step_predictor_gate = StepPredictorGate(
            hidden_dim=hidden_dim,
            num_steps=int(num_steps),
            mlp_hidden_dim=int(mlp_hidden_dim),
            dropout=float(dropout),
        )
        # keep alias for backward compatibility
        self.hard_gate = self.step_predictor_gate
        print(
            f"StepPredictorGate initialized: hidden_dim={hidden_dim}, "
            f"num_steps={int(num_steps)}, mlp_hidden_dim={int(mlp_hidden_dim)}, "
            f"dropout={float(dropout)}"
        )

    def setup_hard_gate(
        self,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        force_reinit: bool = False,
    ) -> None:
        """Backward-compatible wrapper. Internally uses StepPredictorGate."""
        self.setup_step_predictor_gate(
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            force_reinit=force_reinit,
            num_steps=self._num_thought_tokens,
        )

    def clear_hard_gate_selection_history(self) -> None:
        self._hard_gate_selection_history = []

    def get_hard_gate_selection_history(self) -> List[List[int]]:
        return list(self._hard_gate_selection_history)

    def compute_step_predictor_logits(
        self,
        thought_embeddings: Union[List[torch.Tensor], torch.Tensor],
        query_context_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        由所有 thought token（可选融合 query-context）预测步骤 logits。

        Args:
            thought_embeddings: [B, K, D] 或长度为 K 的 List([B, D])
            query_context_embeddings: [B, D]，query 部分编码后的语义向量。
                若为 None，则退化为 thought token 均值上下文。

        Returns:
            step_logits: [B, K]
        """
        if self.step_predictor_gate is None:
            raise RuntimeError(
                "StepPredictorGate is not initialized. Call setup_step_predictor_gate() first."
            )

        if isinstance(thought_embeddings, list):
            if len(thought_embeddings) == 0:
                raise ValueError("thought_embeddings list is empty")
            thought_tensor = torch.stack(thought_embeddings, dim=1)  # [B, K, D]
        else:
            thought_tensor = thought_embeddings

        if thought_tensor.dim() != 3:
            raise ValueError(
                f"Expected thought tensor shape [B, K, D], got {tuple(thought_tensor.shape)}"
            )

        step_logits = self.step_predictor_gate(
            thought_tensor,
            query_context_embeddings=query_context_embeddings,
        )

        # 维度一致性检查（K 必须与 gate 输出类别数一致）
        if step_logits.shape[1] != thought_tensor.shape[1]:
            raise ValueError(
                f"StepPredictorGate output dim mismatch: logits K={step_logits.shape[1]} "
                f"but thought tensor K={thought_tensor.shape[1]}."
            )

        return step_logits

    def compute_hard_gate_logits(
        self,
        thought_embeddings: Union[List[torch.Tensor], torch.Tensor],
        query_context_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Backward-compatible alias for step predictor logits."""
        return self.compute_step_predictor_logits(
            thought_embeddings,
            query_context_embeddings=query_context_embeddings,
        )

    def select_best_thought_embedding(
        self,
        thought_embeddings: Union[List[torch.Tensor], torch.Tensor],
        query_context_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select one thought embedding per sample.

        - 训练模式：使用 Gumbel-Softmax(hard=True) 得到 one-hot 权重，
          通过加权求和提取单步 embedding，保留对 gate logits 的梯度。
        - 推理模式：使用 argmax 硬路由，直接索引提取 embedding。

        Returns:
            selected_embeddings: [batch, hidden]
            hard_gate_logits: [batch, K]
            selected_indices: [batch]
        """
        if isinstance(thought_embeddings, list):
            if len(thought_embeddings) == 0:
                raise ValueError("thought_embeddings list is empty")
            thought_tensor = torch.stack(thought_embeddings, dim=1)
        else:
            thought_tensor = thought_embeddings

        # step_logits: [B, K]
        step_logits = self.compute_step_predictor_logits(
            thought_tensor,
            query_context_embeddings=query_context_embeddings,
        )

        if self.training:
            # 训练阶段：Gumbel-Softmax 采样 one-hot 权重，形状 [B, K]
            weights = F.gumbel_softmax(step_logits, tau=1.0, hard=True, dim=-1)

            # 扩展到 [B, K, 1] 后与 thought_tokens [B, K, D] 广播相乘
            # 再在 K 维求和，得到 selected_embeddings [B, D]
            selected_embeddings = (weights.unsqueeze(-1) * thought_tensor).sum(dim=1)

            # 记录被采样/路由到的步骤索引 [B]
            selected_indices = weights.argmax(dim=-1)
        else:
            # 推理阶段：直接 hard argmax 路由
            selected_indices = step_logits.argmax(dim=1)
            batch_idx = torch.arange(thought_tensor.shape[0], device=thought_tensor.device)
            selected_embeddings = thought_tensor[batch_idx, selected_indices]

        selected_embeddings = torch.nn.functional.normalize(selected_embeddings, dim=-1)
        return selected_embeddings, step_logits, selected_indices

    def _debug_dump_encoding_inputs(
        self,
        *,
        source: str,
        task_label: Union[str, List[str], None],
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        num_thought_tokens: int = 0,
        thought_pooling_mode: Optional[str] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print detailed model-input debug info right before encoding forward."""
        if not getattr(self, "debug_input_dump", False):
            return

        print("\n" + "=" * 88)
        print("[JinaV4 DEBUG] Model encoding input dump")
        print(f"[JinaV4 DEBUG] source={source}")
        print(f"[JinaV4 DEBUG] task_label={task_label}")
        print(f"[JinaV4 DEBUG] num_thought_tokens={num_thought_tokens}")
        print(f"[JinaV4 DEBUG] thought_pooling_mode={thought_pooling_mode}")

        if input_ids is not None:
            print(
                f"[JinaV4 DEBUG] input_ids.shape={tuple(input_ids.shape)} "
                f"dtype={input_ids.dtype} device={input_ids.device}"
            )
        else:
            print("[JinaV4 DEBUG] input_ids=None")

        if attention_mask is not None:
            print(
                f"[JinaV4 DEBUG] attention_mask.shape={tuple(attention_mask.shape)} "
                f"dtype={attention_mask.dtype} device={attention_mask.device}"
            )
        else:
            print("[JinaV4 DEBUG] attention_mask=None")

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    print(
                        f"[JinaV4 DEBUG] kwarg[{k}].shape={tuple(v.shape)} "
                        f"dtype={v.dtype} device={v.device}"
                    )
                else:
                    print(f"[JinaV4 DEBUG] kwarg[{k}]={type(v)}")

        if input_ids is not None:
            max_rows = min(int(getattr(self, "debug_input_max_rows", 4)), int(input_ids.shape[0]))
            tokenizer = getattr(getattr(self, "processor", None), "tokenizer", None)
            for i in range(max_rows):
                row_ids = input_ids[i].detach().cpu().tolist()
                print(f"[JinaV4 DEBUG] row={i} input_ids={row_ids}")
                if attention_mask is not None and attention_mask.dim() >= 2:
                    row_mask = attention_mask[i].detach().cpu().tolist()
                    print(f"[JinaV4 DEBUG] row={i} attention_mask={row_mask}")
                if tokenizer is not None:
                    try:
                        decoded = tokenizer.decode(row_ids, skip_special_tokens=False)
                        print(f"[JinaV4 DEBUG] row={i} decoded={decoded}")
                    except Exception as e:
                        print(f"[JinaV4 DEBUG] row={i} decode_failed={e}")

        print("=" * 88 + "\n")

    def setup_thought_tokens(self, num_thought_tokens: int, semantic_init_token: str = ".", skip_init: bool = False):
        """
        Set up thought tokens for reasoning with semantic initialization.
        
        This method should be called before training/inference when using
        thought tokens for multi-step reasoning.
        
        Key features:
        1. Each <thought_i> gets a unique Token ID (enforced by tokenizer)
        2. Semantic initialization: Initialize from a meaningful token (e.g., ".") 
           with small orthogonal perturbation to help faster convergence
        3. The thought token embeddings will be made trainable separately
        
        Args:
            num_thought_tokens: Number of thought tokens (K in the design doc)
            semantic_init_token: Token to use for semantic initialization (default: ".")
            skip_init: If True, skip semantic initialization (use when loading from checkpoint)
        """
        if num_thought_tokens <= 0:
            return
            
        self._num_thought_tokens = num_thought_tokens
        
        # Add tokens to processor/tokenizer
        self._thought_token_ids = self.processor.add_thought_tokens(num_thought_tokens)
        
        # Resize token embeddings if needed
        # Note: Qwen2_5_VLModel uses self.model.language_model.embed_tokens, not self.model.embed_tokens
        current_vocab_size = self.model.language_model.embed_tokens.weight.shape[0]
        new_vocab_size = len(self.processor.tokenizer)
        
        if new_vocab_size > current_vocab_size:
            print(f"Resizing token embeddings from {current_vocab_size} to {new_vocab_size}")
            self.resize_token_embeddings(new_vocab_size)
            
            # ============ Semantic Initialization ============
            # Instead of random initialization, initialize thought tokens from 
            # a meaningful token with small perturbation:
            # E(<thought_i>) = E(semantic_token) + N(0, epsilon)
            if not skip_init:
                self._semantic_init_thought_tokens(semantic_init_token)
            else:
                print("Skipping semantic initialization (will load from checkpoint)")
            
        print(f"Thought tokens set up: {num_thought_tokens} tokens with IDs {self._thought_token_ids}")
    
    def _semantic_init_thought_tokens(self, semantic_init_token: str = ".", epsilon: float = 0.01):
        """
        Initialize thought token embeddings with semantic initialization.
        
        Instead of random noise, initialize from a meaningful token (like ".")
        with small orthogonal perturbation. This helps the model understand
        these tokens are for "thinking" from the start, saving ~500 steps of 
        exploration.
        
        Formula: E(<thought_i>) = E(semantic_token) + N(0, epsilon)
        
        Args:
            semantic_init_token: Base token for initialization (default "." as a neutral token)
            epsilon: Standard deviation for perturbation (default 0.01)
        """
        if self._thought_token_ids is None or len(self._thought_token_ids) == 0:
            return
            
        # Get the embedding layer
        # Note: Qwen2_5_VLModel uses self.model.language_model.embed_tokens
        embed_layer = self.model.language_model.embed_tokens
        
        # Get the base token's embedding
        # Try multiple candidates in case one doesn't exist
        candidate_tokens = [semantic_init_token, ".", ",", "the", "a"]
        base_embedding = None
        
        for token in candidate_tokens:
            token_ids = self.processor.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) > 0:
                base_token_id = token_ids[0]
                if base_token_id < embed_layer.weight.shape[0]:
                    base_embedding = embed_layer.weight.data[base_token_id].clone()
                    print(f"Using token '{token}' (ID={base_token_id}) for semantic initialization")
                    break
        
        if base_embedding is None:
            # Fallback: use mean of all embeddings
            base_embedding = embed_layer.weight.data.mean(dim=0)
            print("Using mean embedding for semantic initialization (fallback)")
        
        # Initialize each thought token with base + small perturbation
        with torch.no_grad():
            for i, token_id in enumerate(self._thought_token_ids):
                # Add small Gaussian perturbation for each token to make them slightly different
                perturbation = torch.randn_like(base_embedding) * epsilon
                embed_layer.weight.data[token_id] = base_embedding + perturbation
                
        print(f"Semantic initialization complete: {len(self._thought_token_ids)} thought tokens "
              f"initialized with epsilon={epsilon}")

    @property
    def task(self) -> Optional[str]:
        """Get the current task set for the model."""
        return self._task

    @task.setter
    def task(self, task: str):
        """
        Set the task for the model.
        Args:
            task (str): The task name. Must be one of ['retrieval', 'text-matching', 'code']
        """
        if task not in self.config.task_names:
            raise ValueError(
                f"Invalid task: {task}. Must be one of {self.config.task_names}."
            )
        self._task = task

    def get_last_hidden_states(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)], dim=0
            )
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=kwargs.get("image_grid_thw", None),
            attention_mask=attention_mask,
        )

        kwargs["output_hidden_states"] = True
        outputs = super().forward(
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise ValueError("Hidden states not found in model output")

        return hidden_states[-1]

    def _init_projection_layer(self, config) -> None:
        """
        Initializes projection layers.
        """
        self.config.multi_vector_projector_dim = config.multi_vector_projector_dim

        self.multi_vector_projector = nn.Linear(
            in_features=self.config.text_config.hidden_size,
            out_features=self.config.multi_vector_projector_dim,
        )

    def get_thought_token_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        num_thought_tokens: int,
    ) -> List[torch.Tensor]:
        """
        Extract embeddings at thought token positions.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len]
            num_thought_tokens: Number of thought tokens to extract

        Returns:
            List of normalized embeddings [batch_size, hidden_size], one for each thought token
        """
        if num_thought_tokens <= 0 or self._thought_token_ids is None:
            return []

        if len(self._thought_token_ids) < num_thought_tokens:
            raise ValueError(
                f"Requested {num_thought_tokens} thought tokens, but only "
                f"{len(self._thought_token_ids)} are initialized. "
                "Call setup_thought_tokens() first."
            )

        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        thought_embeddings = []

        for i in range(num_thought_tokens):
            token_id = self._thought_token_ids[i]
            token_mask = input_ids == token_id
            if not torch.all(token_mask.any(dim=1)):
                missing_rows = (~token_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(
                    f"Thought token {get_thought_token(i + 1)} (ID={token_id}) not found in "
                    f"batch rows {missing_rows}. Input was likely truncated before the "
                    "thought-token suffix or setup_thought_tokens() was not applied consistently."
                )

            # Find positions of this thought token in each batch item
            # Shape: [batch_size]
            positions = token_mask.long().argmax(dim=1)

            # Extract embeddings at these positions
            # Using advanced indexing: hidden_states[batch_idx, position_idx]
            batch_indices = torch.arange(batch_size, device=device)
            embeddings = hidden_states[batch_indices, positions]  # [batch_size, hidden_size]

            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            thought_embeddings.append(embeddings)

        return thought_embeddings

    def get_query_context_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        num_thought_tokens: int,
    ) -> torch.Tensor:
        """
        Build query-context embedding from encoded sequence states.

        When thought tokens are present, this pools only tokens before <thought_1>.
        If no valid pre-thought token exists for a sample, it falls back to
        mean-pooling over all valid tokens.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected hidden_states shape [B, L, D], got {tuple(hidden_states.shape)}"
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.long,
            )

        valid_mask = attention_mask.bool()
        query_mask = valid_mask.clone()

        if (
            num_thought_tokens > 0
            and self._thought_token_ids is not None
            and len(self._thought_token_ids) >= 1
        ):
            first_thought_id = self._thought_token_ids[0]
            first_mask = input_ids == first_thought_id
            has_first = first_mask.any(dim=1)
            first_pos = first_mask.long().argmax(dim=1)

            seq_len = hidden_states.shape[1]
            seq_idx = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            before_first = seq_idx < first_pos.unsqueeze(1)
            query_mask = valid_mask & (before_first | (~has_first).unsqueeze(1))

        # Fallback to full valid sequence if a sample has zero pre-thought tokens.
        empty_rows = ~query_mask.any(dim=1)
        if empty_rows.any():
            query_mask = query_mask.clone()
            query_mask[empty_rows] = valid_mask[empty_rows]

        query_mask_f = query_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * query_mask_f).sum(dim=1) / query_mask_f.sum(dim=1).clamp_min(1.0)
        pooled = F.normalize(pooled, dim=-1)
        return pooled

    def get_single_vector_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        num_thought_tokens: int = 0,
        thought_pooling_mode: str = "last",
    ) -> torch.Tensor:
        """
        Get the single-vector embeddings from the hidden states.

        When using thought tokens (num_thought_tokens > 0), supports two modes:
        1) "last": returns embedding at <thought_K>
        2) "mean_all_thought_tokens": mean-pools embeddings from <thought_1>...<thought_K>
        Otherwise, falls back to mean pooling over attention mask.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            input_ids: [batch_size, seq_len] - required when using thought tokens
            num_thought_tokens: Number of thought tokens (K). If > 0, use last thought token.
            thought_pooling_mode: Pooling mode for thought-token final embedding.

        Returns:
            Normalized embeddings [batch_size, hidden_size]
        """
        has_initialized_thought_tokens = (
            self._thought_token_ids is not None and len(self._thought_token_ids) >= num_thought_tokens
        )
        # print(f"Getting single-vector embeddings with num_thought_tokens={num_thought_tokens}, "
        #       f"thought_pooling_mode={thought_pooling_mode}, "
        #       f"has_initialized_thought_tokens={has_initialized_thought_tokens}")
        if num_thought_tokens > 0:
            if input_ids is None:
                raise ValueError(
                    "input_ids is required when num_thought_tokens > 0 to extract query embeddings "
                    "from thought token positions."
                )
            if not has_initialized_thought_tokens:
                raise ValueError(
                    f"num_thought_tokens={num_thought_tokens}, but thought token IDs are not initialized. "
                    "Call setup_thought_tokens() first."
                )

            thought_embeddings = self.get_thought_token_embeddings(
                hidden_states=hidden_states,
                input_ids=input_ids,
                num_thought_tokens=num_thought_tokens,
            )

            if thought_pooling_mode == "mean_all_thought_tokens":
                #print("Using 'mean_all_thought_tokens' pooling mode: mean-pooling embeddings from all thought tokens as the final representation.")
                # Mean-pool all thought token embeddings as the final representation
                pooled_output = torch.stack(thought_embeddings, dim=0).mean(dim=0)
            elif thought_pooling_mode == "last":
                #print("Using 'last' pooling mode: returning embedding from the last thought token as the final representation.")
                # Use the last thought token's embedding as the final representation
                pooled_output = thought_embeddings[-1]
            else:
                raise ValueError(
                    f"Unsupported thought_pooling_mode: {thought_pooling_mode}. "
                    f"Expected one of ['last', 'mean_all_thought_tokens']"
                )
        else:
            # print("NOWarning: num_thought_tokens=0, falling back to mean pooling over attention mask for single-vector embedding.")
            # Fall back to mean pooling
            pooled_output = torch.sum(
                hidden_states * attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)

        return torch.nn.functional.normalize(pooled_output, dim=-1)


    def get_multi_vector_embeddings(
        self,
        task_label: Union[str, List[str]],
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project the hidden states to multi-vector embeddings.
        """
        multi_vec_emb = self.multi_vector_projector(
            hidden_states, task_label=task_label
        )
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)
        return multi_vec_emb * attention_mask.unsqueeze(-1)

    def _input_has_image(self, input_ids):
        return self.config.vision_start_token_id in input_ids

    ### ==================== MBEIR Interface Methods ====================
    
    def get_img_preprocess_fn(self):
        """
        Returns image preprocessing function that preserves raw pixel values.
        """
        from torchvision import transforms

        def img_preprocess_wrapper(image: Image.Image) -> torch.Tensor:
            target_size = getattr(self, 'mbeir_image_size', (224, 224))
            if isinstance(target_size, int):
                target_size = (target_size, target_size)

            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])
            return transform(image)

        return img_preprocess_wrapper

    def get_tokenizer(self):
        """
        Returns a passthrough function that preserves raw text strings.
        """
        def passthrough_wrapper(texts: List[str]) -> RawTextBatch:
            return RawTextBatch(texts)

        return passthrough_wrapper

    def encode_mbeir_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode a batch from UniIR MBEIR dataset using Jina V4's native processing.
        """
        is_query_batch = "qid_list" in batch and batch["qid_list"] is not None
        is_candidate_batch = "did_list" in batch and batch["did_list"] is not None
        id_list = batch.get("did_list") or batch.get("qid_list")
        if id_list is None:
            raise ValueError("id_list (did_list or qid_list) not found in batch.")

        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask = batch["txt_mask_batched"]
        image_mask = batch["image_mask_batched"]

        batch_size = image_batched.size(0)
        device = image_batched.device
        task_label = getattr(self, 'mbeir_task_label', 'retrieval')
        # Encoding mode:
        # - asymmetric (default): only queries use thought tokens
        # - symmetric: both queries and candidates use thought tokens
        symmetric_qc_encoding = bool(getattr(self, 'mbeir_symmetric_encoding', False))
        if is_query_batch and not is_candidate_batch:
            num_thought_tokens = getattr(self, 'mbeir_num_thought_tokens', 0)
        elif is_candidate_batch and not is_query_batch:
            num_thought_tokens = getattr(self, 'mbeir_num_thought_tokens', 0) if symmetric_qc_encoding else 0
        else:
            num_thought_tokens = 0

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

        embed_dim = self.config.text_config.hidden_size
        embeddings = torch.zeros(batch_size, embed_dim, device=device)

        # Process each modality group
        if text_only_idx:
            embeddings[text_only_idx] = self._encode_text_batch(
                txt_batched, text_only_idx, task_label, device, num_thought_tokens
            )
        if image_only_idx:
            embeddings[image_only_idx] = self._encode_image_batch(
                image_batched, image_only_idx, task_label, device, num_thought_tokens
            )
        if multimodal_idx:
            embeddings[multimodal_idx] = self._encode_multimodal_batch(
                txt_batched, image_batched, multimodal_idx, task_label, device, num_thought_tokens
            )

        return embeddings, id_list

    def encode_mbeir_batch_at_step(
        self, batch: Dict[str, Any], thought_step: int
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode a batch using a specific thought step's embedding for queries.

        For candidate batches, behaves identically to encode_mbeir_batch.
        For query batches, runs forward with all thought tokens but uses the
        thought_step-th token's embedding instead of the last one.

        This enables truncated reasoning analysis: evaluating retrieval
        performance at each intermediate thought step.

        Args:
            batch: MBEIR batch dict
            thought_step: Which thought token to use as embedding (1-indexed).
                         e.g., thought_step=1 uses <thought_1>'s embedding.
        """
        is_query_batch = "qid_list" in batch and batch["qid_list"] is not None
        is_candidate_batch = "did_list" in batch and batch["did_list"] is not None

        # For candidate batches, delegate to standard encoding (no thought tokens)
        if is_candidate_batch and not is_query_batch:
            return self.encode_mbeir_batch(batch)

        id_list = batch.get("qid_list")
        if id_list is None:
            raise ValueError("qid_list not found in query batch.")

        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask = batch["txt_mask_batched"]
        image_mask = batch["image_mask_batched"]

        batch_size = image_batched.size(0)
        device = image_batched.device
        task_label = getattr(self, 'mbeir_task_label', 'retrieval')
        num_thought_tokens = self._num_thought_tokens

        assert 1 <= thought_step <= num_thought_tokens, \
            f"thought_step must be between 1 and {num_thought_tokens}, got {thought_step}"

        # Group indices by modality (same logic as encode_mbeir_batch)
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

        embed_dim = self.config.text_config.hidden_size
        embeddings = torch.zeros(batch_size, embed_dim, device=device)

        # Process each modality group, extracting the specific thought step
        if text_only_idx:
            texts = txt_batched[text_only_idx]
            processed = self.processor.process_texts(texts, num_thought_tokens=num_thought_tokens)
            processed = {k: v.to(device) for k, v in processed.items()}
            output = self._forward_embeddings(
                task_label=task_label,
                num_thought_tokens=num_thought_tokens,
                return_thought_embeddings=True,
                **processed
            )
            embeddings[text_only_idx] = output.thought_embeddings[thought_step - 1]

        if image_only_idx:
            from torchvision.transforms.functional import to_pil_image
            pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in image_only_idx]
            processed = self.processor.process_images(pil_images, num_thought_tokens=num_thought_tokens)
            processed = {k: v.to(device) for k, v in processed.items()}
            output = self._forward_embeddings(
                task_label=task_label,
                num_thought_tokens=num_thought_tokens,
                return_thought_embeddings=True,
                **processed
            )
            embeddings[image_only_idx] = output.thought_embeddings[thought_step - 1]

        if multimodal_idx:
            from torchvision.transforms.functional import to_pil_image
            pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in multimodal_idx]
            texts = txt_batched[multimodal_idx]
            processed = self.processor.process_multimodal(
                pil_images, texts, num_thought_tokens=num_thought_tokens
            )
            processed = {k: v.to(device) for k, v in processed.items()}
            output = self._forward_embeddings(
                task_label=task_label,
                num_thought_tokens=num_thought_tokens,
                return_thought_embeddings=True,
                **processed
            )
            embeddings[multimodal_idx] = output.thought_embeddings[thought_step - 1]

        return embeddings, id_list

    def encode_mbeir_batch_all_steps(
        self, batch: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Encode a batch and return embeddings at ALL thought token positions
        in a single forward pass.

        For candidate batches, delegates to standard encode_mbeir_batch.
        For query batches, extracts all K thought token embeddings simultaneously.

        Returns:
            (all_step_embeddings, id_list) where all_step_embeddings is a list
            of K tensors, each of shape (batch_size, embed_dim).
            The i-th tensor corresponds to <thought_{i+1}>'s embedding.
        """
        is_query_batch = "qid_list" in batch and batch["qid_list"] is not None
        is_candidate_batch = "did_list" in batch and batch["did_list"] is not None

        if is_candidate_batch and not is_query_batch:
            emb, ids = self.encode_mbeir_batch(batch)
            return [emb], ids

        id_list = batch.get("qid_list")
        if id_list is None:
            raise ValueError("qid_list not found in query batch.")

        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask = batch["txt_mask_batched"]
        image_mask = batch["image_mask_batched"]

        batch_size = image_batched.size(0)
        device = image_batched.device
        task_label = getattr(self, 'mbeir_task_label', 'retrieval')
        num_thought_tokens = self._num_thought_tokens

        embed_dim = self.config.text_config.hidden_size
        all_step_embeddings = [
            torch.zeros(batch_size, embed_dim, device=device)
            for _ in range(num_thought_tokens)
        ]

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

        if text_only_idx:
            texts = txt_batched[text_only_idx]
            processed = self.processor.process_texts(texts, num_thought_tokens=num_thought_tokens)
            processed = {k: v.to(device) for k, v in processed.items()}
            output = self._forward_embeddings(
                task_label=task_label,
                num_thought_tokens=num_thought_tokens,
                return_thought_embeddings=True,
                **processed
            )
            for step_idx in range(num_thought_tokens):
                all_step_embeddings[step_idx][text_only_idx] = output.thought_embeddings[step_idx]

        if image_only_idx:
            from torchvision.transforms.functional import to_pil_image
            pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in image_only_idx]
            processed = self.processor.process_images(pil_images, num_thought_tokens=num_thought_tokens)
            processed = {k: v.to(device) for k, v in processed.items()}
            output = self._forward_embeddings(
                task_label=task_label,
                num_thought_tokens=num_thought_tokens,
                return_thought_embeddings=True,
                **processed
            )
            for step_idx in range(num_thought_tokens):
                all_step_embeddings[step_idx][image_only_idx] = output.thought_embeddings[step_idx]

        if multimodal_idx:
            from torchvision.transforms.functional import to_pil_image
            pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in multimodal_idx]
            texts = txt_batched[multimodal_idx]
            processed = self.processor.process_multimodal(
                pil_images, texts, num_thought_tokens=num_thought_tokens
            )
            processed = {k: v.to(device) for k, v in processed.items()}
            output = self._forward_embeddings(
                task_label=task_label,
                num_thought_tokens=num_thought_tokens,
                return_thought_embeddings=True,
                **processed
            )
            for step_idx in range(num_thought_tokens):
                all_step_embeddings[step_idx][multimodal_idx] = output.thought_embeddings[step_idx]

        return all_step_embeddings, id_list

    def _encode_text_batch(
        self,
        txt_batched,
        indices: List[int],
        task_label: str,
        device: torch.device,
        num_thought_tokens: int = 0,
    ) -> torch.Tensor:
        """Encode text-only samples using Jina V4's native process_texts."""
        texts = txt_batched[indices]
        processed = self.processor.process_texts(texts, num_thought_tokens=num_thought_tokens)
        processed = {k: v.to(device) for k, v in processed.items()}
        thought_pooling_mode = getattr(self, 'mbeir_thought_pooling_mode', getattr(self, 'thought_pooling_mode', 'last'))
        output = self._forward_embeddings(
            task_label=task_label, 
            num_thought_tokens=num_thought_tokens, 
            thought_pooling_mode=thought_pooling_mode,
            **processed
        )
        return output.single_vec_emb

    def _encode_image_batch(
        self,
        image_batched: torch.Tensor,
        indices: List[int],
        task_label: str,
        device: torch.device,
        num_thought_tokens: int = 0,
    ) -> torch.Tensor:
        """Encode image-only samples using Jina V4's native process_images."""
        from torchvision.transforms.functional import to_pil_image
        pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in indices]
        processed = self.processor.process_images(pil_images, num_thought_tokens=num_thought_tokens)
        processed = {k: v.to(device) for k, v in processed.items()}
        thought_pooling_mode = getattr(self, 'mbeir_thought_pooling_mode', getattr(self, 'thought_pooling_mode', 'last'))
        output = self._forward_embeddings(
            task_label=task_label, 
            num_thought_tokens=num_thought_tokens, 
            thought_pooling_mode=thought_pooling_mode,
            **processed
        )
        return output.single_vec_emb

    def _encode_multimodal_batch(
        self,
        txt_batched,
        image_batched: torch.Tensor,
        indices: List[int],
        task_label: str,
        device: torch.device,
        num_thought_tokens: int = 0,
    ) -> torch.Tensor:
        """Encode multimodal samples using Jina V4's native process_multimodal."""
        from torchvision.transforms.functional import to_pil_image
        pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in indices]
        texts = txt_batched[indices]
        processed = self.processor.process_multimodal(
            pil_images, texts, num_thought_tokens=num_thought_tokens
        )
        processed = {k: v.to(device) for k, v in processed.items()}
        thought_pooling_mode = getattr(self, 'mbeir_thought_pooling_mode', getattr(self, 'thought_pooling_mode', 'last'))
        output = self._forward_embeddings(
            task_label=task_label, 
            num_thought_tokens=num_thought_tokens, 
            thought_pooling_mode=thought_pooling_mode,
            **processed
        )
        return output.single_vec_emb

    def _forward_embeddings(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        num_thought_tokens: int = 0,
        thought_pooling_mode: Optional[str] = None,
        return_thought_embeddings: bool = False,
        **kwargs,
    ) -> JinaEmbeddingsV4ModelOutput:
        """
        Internal forward pass for embedding generation with thought token support.
        
        Args:
            task_label: Task identifier for LoRA adapter selection
            input_ids: Input token IDs (includes thought tokens if num_thought_tokens > 0)
            attention_mask: Attention mask
            num_thought_tokens: Number of thought tokens appended to input (K)
            thought_pooling_mode: Final embedding mode when thought tokens are used.
            return_thought_embeddings: If True, return embeddings at each thought token position
            **kwargs: Additional arguments (pixel_values, etc.)
            
        Returns:
            JinaEmbeddingsV4ModelOutput with:
                - single_vec_emb: Embedding from the last thought token (or mean pooling if no thought tokens)
                - thought_embeddings: List of embeddings from each thought token (if return_thought_embeddings=True)
        """
        self._debug_dump_encoding_inputs(
            source="_forward_embeddings",
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_thought_tokens=num_thought_tokens,
            thought_pooling_mode=thought_pooling_mode,
            kwargs=kwargs,
        )

        # Single forward pass
        hidden_states = self.get_last_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_label=task_label,
            **kwargs,
        )

        effective_pooling_mode = thought_pooling_mode or getattr(self, 'thought_pooling_mode', 'last')

        # Extract thought token embeddings when needed:
        # - deep supervision output
        # - hard-gate selection mode
        need_thought_embeddings = (
            num_thought_tokens > 0 and (
                return_thought_embeddings or effective_pooling_mode == "hard_gate"
            )
        )

        thought_embeddings = None
        if need_thought_embeddings:
            thought_embeddings = self.get_thought_token_embeddings(
                hidden_states=hidden_states,
                input_ids=input_ids,
                num_thought_tokens=num_thought_tokens,
            )

        hard_gate_logits = None
        selected_thought_indices = None
        if effective_pooling_mode == "hard_gate" and num_thought_tokens > 0:
            if thought_embeddings is None:
                raise ValueError("Hard-gate mode requires thought embeddings, but none were extracted.")
            query_context_emb = self.get_query_context_embeddings(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_thought_tokens=num_thought_tokens,
            )
            single_vec_emb, hard_gate_logits, selected_thought_indices = self.select_best_thought_embedding(
                thought_embeddings,
                query_context_embeddings=query_context_emb,
            )
            if not self.training and selected_thought_indices is not None:
                self._hard_gate_selection_history.append(
                    selected_thought_indices.detach().cpu().tolist()
                )
        else:
            # Compute the single-vector embedding (from last thought token or mean pooling)
            single_vec_emb = self.get_single_vector_embeddings(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                input_ids=input_ids,
                num_thought_tokens=num_thought_tokens,
                thought_pooling_mode=effective_pooling_mode,
            )
        
        # Compute multi-vector embeddings
        multi_vec_emb = self.get_multi_vector_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            task_label=task_label,
        )

        return JinaEmbeddingsV4ModelOutput(
            vlm_last_hidden_states=None,
            single_vec_emb=single_vec_emb,
            multi_vec_emb=multi_vec_emb,
            thought_embeddings=thought_embeddings if return_thought_embeddings else None,
            hard_gate_logits=hard_gate_logits,
            selected_thought_indices=selected_thought_indices,
        )

    ### ==================== End MBEIR Interface Methods ====================

    def forward(
        self,
        task_label: Union[str, List[str]] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        output_vlm_last_hidden_states: bool = False,
        encode_mbeir_batch: bool = False,
        num_thought_tokens: int = 0,
        thought_pooling_mode: Optional[str] = None,
        return_thought_embeddings: bool = False,
        **kwargs,
    ) -> Union[JinaEmbeddingsV4ModelOutput, Tuple[torch.Tensor, List[int]]]:
        """
        Forward pass through the model. Returns both single-vector and multi-vector embeddings.
        
        Args:
            task_label: Task identifier for LoRA adapter selection
            input_ids (torch.Tensor): The input tokens tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            output_vlm_last_hidden_states: Whether to return hidden states
            encode_mbeir_batch: If True, use MBEIR batch encoding mode
            num_thought_tokens: Number of thought tokens (K). When > 0, uses thought token mechanism.
            thought_pooling_mode: Final embedding mode when thought tokens are used.
            return_thought_embeddings: If True, return embeddings at each thought token position
                         (for deep supervision during training).
                         
        Returns:
            JinaEmbeddingsV4ModelOutput with:
                - single_vec_emb: Embedding from the last thought token
                - multi_vec_emb: Multi-vector embeddings  
                - vlm_last_hidden_states: Hidden states (if output_vlm_last_hidden_states=True)
                - thought_embeddings: List of embeddings from each thought token 
                    (if return_thought_embeddings=True)
            Or (embeddings, id_list) if encode_mbeir_batch=True
        """
        # Handle MBEIR batch encoding mode
        if encode_mbeir_batch:
            return self.encode_mbeir_batch(kwargs)

        self._debug_dump_encoding_inputs(
            source="forward",
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_thought_tokens=num_thought_tokens,
            thought_pooling_mode=thought_pooling_mode,
            kwargs=kwargs,
        )

        # Single forward pass
        hidden_states = self.get_last_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_label=task_label,
            **kwargs,
        )

        effective_pooling_mode = thought_pooling_mode or getattr(self, 'thought_pooling_mode', 'last')

        # Extract thought token embeddings when needed
        need_thought_embeddings = (
            num_thought_tokens > 0 and (
                return_thought_embeddings or effective_pooling_mode == "hard_gate"
            )
        )

        thought_embeddings = None
        if need_thought_embeddings:
            thought_embeddings = self.get_thought_token_embeddings(
                hidden_states=hidden_states,
                input_ids=input_ids,
                num_thought_tokens=num_thought_tokens,
            )

        hard_gate_logits = None
        selected_thought_indices = None
        if effective_pooling_mode == "hard_gate" and num_thought_tokens > 0:
            if thought_embeddings is None:
                raise ValueError("Hard-gate mode requires thought embeddings, but none were extracted.")
            query_context_emb = self.get_query_context_embeddings(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_thought_tokens=num_thought_tokens,
            )
            single_vec_emb, hard_gate_logits, selected_thought_indices = self.select_best_thought_embedding(
                thought_embeddings,
                query_context_embeddings=query_context_emb,
            )
            if not self.training and selected_thought_indices is not None:
                self._hard_gate_selection_history.append(
                    selected_thought_indices.detach().cpu().tolist()
                )
        else:
            # Compute the embeddings
            single_vec_emb = self.get_single_vector_embeddings(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                input_ids=input_ids,
                num_thought_tokens=num_thought_tokens,
                thought_pooling_mode=effective_pooling_mode,
            )
        multi_vec_emb = self.get_multi_vector_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            task_label=task_label,
        )

        return JinaEmbeddingsV4ModelOutput(
            vlm_last_hidden_states=(
                hidden_states if output_vlm_last_hidden_states else None
            ),
            single_vec_emb=single_vec_emb,
            multi_vec_emb=multi_vec_emb,
            thought_embeddings=thought_embeddings if return_thought_embeddings else None,
            hard_gate_logits=hard_gate_logits,
            selected_thought_indices=selected_thought_indices,
        )

    def _process_batches(
        self,
        data: List[Union[str, Image.Image]],
        task_label: Union[str, List[str]],
        processor_fn: Callable,
        desc: str,
        return_multivector: bool = False,
        return_numpy: bool = False,
        batch_size: int = 32,
        truncate_dim: Optional[int] = None,
        num_thought_tokens: int = 0,
    ) -> Union[np.ndarray, List[torch.Tensor]]:
        dataloader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=processor_fn,
        )
        if return_multivector and len(data) > 1:
            assert (
                not return_numpy
            ), "`return_numpy` is not supported when `return_multivector=True` and more than one data is encoded"
        results = []
        self.eval()
        for batch in tqdm(dataloader, desc=desc, disable=self.verbosity == 0):
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(
                    device_type=torch.device(self.device).type, dtype=torch.bfloat16
                ):
                    embeddings = self(**batch, task_label=task_label, num_thought_tokens=num_thought_tokens)
                    if not return_multivector:
                        embeddings = embeddings.single_vec_emb
                        if truncate_dim is not None:
                            embeddings = embeddings[:, :truncate_dim]
                            embeddings = torch.nn.functional.normalize(
                                embeddings, p=2, dim=-1
                            )
                    else:
                        embeddings = embeddings.multi_vec_emb

                    if return_multivector and not return_numpy:
                        valid_tokens = batch["attention_mask"].bool()
                        embeddings = [
                            emb[mask] for emb, mask in zip(embeddings, valid_tokens)
                        ]
                        results.append(embeddings)
                    else:
                        results.append(
                            embeddings.cpu()
                            if return_numpy
                            else list(torch.unbind(embeddings))
                        )
        if return_numpy:
            return np.concatenate([result.numpy() for result in results], axis=0)
        return [item for sublist in results for item in sublist]

    def _validate_encoding_params(
        self,
        truncate_dim: Optional[int] = None,
        prompt_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        encode_kwargs = {}
        if prompt_name is not None:
            if prompt_name not in PREFIX_DICT:
                raise ValueError(
                    f"Invalid prompt_name: {prompt_name}. Must be one of {list(PREFIX_DICT.keys())}."
                )
            else:
                encode_kwargs["prefix"] = (
                    PREFIX_DICT[prompt_name]
                    if self.task != "text-matching"
                    else PREFIX_DICT["query"]
                )

        truncate_dim = truncate_dim or self.config.truncate_dim
        if truncate_dim is not None and truncate_dim not in self.config.matryoshka_dims:
            raise ValueError(
                f"Invalid truncate_dim: {truncate_dim}. Must be one of {self.config.matryoshka_dims}."
            )
        else:
            encode_kwargs["truncate_dim"] = truncate_dim

        return encode_kwargs

    def _validate_task(self, task: Optional[str] = None) -> str:
        if task is None:
            if self.task is None:
                raise ValueError(
                    "Task must be specified before encoding data. You can set it either as a model property "
                    "(e.g., model.task = 'retrieval') or pass it as an argument to the encode method."
                )
            task = self.task
        else:
            if task not in self.config.task_names:
                raise ValueError(
                    f"Invalid task: {task}. Must be one of {self.config.task_names}."
                )
        return task

    def encode_text(
        self,
        texts: Union[str, List[str]],
        task: Optional[str] = None,
        max_length: int = 32768,
        batch_size: int = 8,
        return_multivector: bool = False,
        return_numpy: bool = False,
        truncate_dim: Optional[int] = None,
        prompt_name: Optional[str] = None,
        num_thought_tokens: int = 0,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Encodes a list of texts into embeddings.
        """
        prompt_name = prompt_name or "query"
        encode_kwargs = self._validate_encoding_params(
            truncate_dim=truncate_dim, prompt_name=prompt_name
        )

        task = self._validate_task(task)

        processor_fn = partial(
            self.processor.process_texts,
            max_length=max_length,
            prefix=encode_kwargs.pop("prefix"),
            num_thought_tokens=num_thought_tokens,
        )

        return_list = isinstance(texts, list)

        if return_multivector and return_list and len(texts) > 1:
            if return_numpy:
                print(
                    "Warning: `return_numpy` is ignored when `return_multivector=True` and `len(texts) > 1`"
                )
            return_numpy = False

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._process_batches(
            data=texts,
            processor_fn=processor_fn,
            desc="Encoding texts...",
            task_label=task,
            return_multivector=return_multivector,
            return_numpy=return_numpy,
            batch_size=batch_size,
            num_thought_tokens=num_thought_tokens,
            **encode_kwargs,
        )

        return embeddings if return_list else embeddings[0]

    def _load_images_if_needed(
        self, images: List[Union[str, Image.Image]]
    ) -> List[Image.Image]:
        loaded_images = []
        for image in images:
            if isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    image = Image.open(image).convert("RGB")
            loaded_images.append(image)
        return loaded_images

    def encode_image(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        task: Optional[str] = None,
        batch_size: int = 8,
        return_multivector: bool = False,
        return_numpy: bool = False,
        truncate_dim: Optional[int] = None,
        max_pixels: Optional[int] = None,
        num_thought_tokens: int = 0,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Encodes a list of images or a single image into embedding(s).
        """
        if max_pixels:
            default_max_pixels = self.processor.image_processor.max_pixels
            self.processor.image_processor.max_pixels = max_pixels

        encode_kwargs = self._validate_encoding_params(truncate_dim=truncate_dim)
        task = self._validate_task(task)

        return_list = isinstance(images, list)

        if return_multivector and return_list and len(images) > 1:
            if return_numpy:
                print(
                    "Warning: `return_numpy` is ignored when `return_multivector=True` and `len(images) > 1`"
                )
            return_numpy = False

        if isinstance(images, (str, Image.Image)):
            images = [images]

        images = self._load_images_if_needed(images)
        
        processor_fn = partial(
            self.processor.process_images,
            num_thought_tokens=num_thought_tokens,
        )
        
        embeddings = self._process_batches(
            data=images,
            processor_fn=processor_fn,
            desc="Encoding images...",
            task_label=task,
            batch_size=batch_size,
            return_multivector=return_multivector,
            return_numpy=return_numpy,
            num_thought_tokens=num_thought_tokens,
            **encode_kwargs,
        )

        if max_pixels:
            self.processor.image_processor.max_pixels = default_max_pixels

        return embeddings if return_list else embeddings[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        """
        Loads a pretrained model and configures it with the appropriate task adapter.
        """
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = "auto"

        kwargs["key_mapping"] = super()._checkpoint_conversion_mapping
        if not is_flash_attn_2_available():
            kwargs["attn_implementation"] = "sdpa"

        base_model = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        # Configure adapter directory
        if os.path.isdir(base_model.name_or_path):
            adapter_dir = os.path.join(base_model.name_or_path, "adapters")
        else:
            adapter_cache_path = snapshot_download(
                repo_id=base_model.name_or_path, allow_patterns=["adapters/*"]
            )
            adapter_dir = os.path.join(adapter_cache_path, "adapters")

        lora_config = LoraConfig.from_pretrained(adapter_dir)
        lora_config._custom_modules = {
            torch.nn.modules.linear.Linear: partial(
                MultiAdapterLinear,
                task_names=base_model.config.task_names,
            )
        }
        peft_model = PeftModel.from_pretrained(
            model=base_model,
            model_id=adapter_dir,
            config=lora_config,
        )

        def task_getter(self):
            return self.model.task

        def task_setter(self, value):
            self.model.task = value

        peft_model.__class__.task = property(task_getter, task_setter)

        return peft_model
