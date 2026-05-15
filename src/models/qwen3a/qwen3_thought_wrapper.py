"""
Qwen3-VL Wrapper for MBEIR integration.

This module supports recursive reasoning in embedding space for query encoding:
- Step 0: encode original query and pool e_0
- Step 1..K: append previous step embedding e_{k-1} as an extra token vector,
  run forward again, and pool e_k
- Final representation: normalized e_K

Important: Unlike the original Qwen3VLEmbedder.process() (which is @torch.no_grad()),
this wrapper performs a normal forward pass through a Transformers Qwen3VL model so
that gradients can flow (for LoRA fine-tuning and end-to-end training).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from dataclasses import dataclass

# Ensure required dependencies are importable even if current environment doesn't
# have /data/Qwen3-VL-Embedding in PYTHONPATH.
if '/data/Qwen3-VL-Embedding/src/models' not in sys.path:
    sys.path.insert(0, '/data/Qwen3-VL-Embedding/src/models')

# Reuse the embedding model definition from the external repo.
from qwen3_vl_embedding import Qwen3VLForEmbedding
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from qwen_vl_utils.vision_process import process_vision_info


# Thought token constants
THOUGHT_TOKEN_PREFIX = "<thought_"
FINAL_TOKEN = "<final>"
MAX_THOUGHT_TOKENS = 16


def get_thought_token(idx: int) -> str:
    """Generate thought token string for given index (1-indexed)."""
    return f"{THOUGHT_TOKEN_PREFIX}{idx}>"


def get_all_thought_tokens(num_tokens: int) -> List[str]:
    """Generate list of thought token strings."""
    return [get_thought_token(i) for i in range(1, num_tokens + 1)]


class RawTextBatch:
    """Wrapper to store raw text strings for later processing.
    Must be defined at module level for pickle compatibility with DataLoader's multiprocessing.
    """
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.texts[i] for i in idx]
        return self.texts[idx]


@dataclass
class Qwen3ThoughtOutput:
    """Output structure for Qwen3 with thought tokens."""
    single_vec_emb: torch.Tensor
    thought_embeddings: Optional[List[torch.Tensor]] = None


class ReasoningTokenEmbeddingWrapper(nn.Module):
    """Wrap the model's input embedding layer and override embeddings for
    thought/final special tokens with a small trainable table.

    This avoids making the *entire* vocab embedding matrix trainable (which would
    allocate a dense grad of shape [vocab, hidden] and drastically reduce
    usable batch size).
    """

    def __init__(self, base_embedding: nn.Module, special_token_ids: List[int]):
        super().__init__()
        if not special_token_ids:
            raise ValueError("special_token_ids must be non-empty")

        self.base_embedding = base_embedding
        self.special_token_ids = list(special_token_ids)

        # Match base embedding dtype/device for stability.
        base_weight = getattr(base_embedding, "weight", None)
        if base_weight is None:
            raise ValueError("base_embedding must expose .weight")

        embed_dim = base_weight.shape[1]
        self.special_embedding = nn.Embedding(
            num_embeddings=len(self.special_token_ids),
            embedding_dim=embed_dim,
            device=base_weight.device,
            dtype=base_weight.dtype,
        )

        # 核心改动：将 base_embedding 中已经经过语义初始化(semantic init) 的向量
        # 原封不动地拷贝到 special_embedding 里，防止 fallback 成纯随机向量
        with torch.no_grad():
            for i, tid in enumerate(self.special_token_ids):
                if tid < base_weight.shape[0]:
                    self.special_embedding.weight[i].copy_(base_weight[tid])

    @property
    def weight(self):
        # Keep compatibility with code that reads embedding dtype/shape.
        return self.base_embedding.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # Get base embeddings (non-differentiable if base_embedding is frozen)
        embeds = self.base_embedding(input_ids)
        
        # local_ids: -1 for normal tokens, [0..K] for special tokens
        local_ids = torch.full_like(input_ids, -1)
        for i, tid in enumerate(self.special_token_ids):
            local_ids.masked_fill_(input_ids == tid, i)

        special_pos = local_ids >= 0
        if special_pos.any():
            # Clone to avoid in-place modification of a computation graph leaf
            embeds = embeds.clone()
            
            # Replace the embeddings at the special positions with our trainable ones
            embeds[special_pos] = self.special_embedding(local_ids[special_pos])

        return embeds


class Qwen3VLThoughtWrapper(nn.Module):
    """
    Wrapper for Qwen3-VL with thought token support.

    Key features:
    - setup_thought_tokens(): Add <thought_1..K> + <final> to tokenizer
    - _build_reasoning_attention_mask(): Restrict <final> to only attend thought tokens
    - _forward_embeddings(): Trainable forward with attention mask injection
    - encode_mbeir_batch(): Asymmetric query/doc encoding
    """

    def __init__(
        self,
        model_name_or_path: str = "/data/Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B",
        max_length: int = 1500,
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "sdpa",
        **kwargs
    ):
        super().__init__()
        # Device setup
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_str = f"cuda:{local_rank}"
        target_device_map = {"": device_str}

        # Load model and processor
        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=target_device_map,
            trust_remote_code=True,
            **kwargs
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='right'
        )

        # Keep track of the original attention implementation. We may need to
        # temporarily switch away from FlashAttention2 when injecting a custom
        # 4D attention mask (FlashAttention2 only supports padding/causal masks).
        self._default_attn_implementation = getattr(self.model.config, "_attn_implementation", None)

        self.max_length = max_length
        self.model_name_or_path = model_name_or_path

        # Thought token state
        self._num_thought_tokens = 0
        self._thought_token_ids = None
        self._final_token_id = None
        self.enable_final_token = False

        # MBEIR-specific attributes
        self.mbeir_task_label = 'retrieval'
        self.mbeir_image_size = (224, 224)
        self.mbeir_max_text_length = max_length
        self.mbeir_num_thought_tokens = 0  # Set by config
        self.mbeir_symmetric_encoding = False  # If True, query/candidate both use TT+FT
        self.task = 'retrieval'
        # New default: recursive embedding reasoning (replaces static thought-token concat).
        self.use_recursive_embedding_reasoning = True

        # Debug controls (injected from training config)
        self.debug_mode = False
        self.debug_max_text_chars = 120
        self.debug_preprocess_print_freq = 1
        self.debug_token_preview_len = 12
        self._debug_preprocess_calls = 0

    def _debug_log(self, message: str):
        if bool(getattr(self, "debug_mode", False)):
            print(f"[DEBUG][preprocess] {message}")

    def setup_thought_tokens(
        self,
        num_thought_tokens: int,
        semantic_init_token: str = ".",
        skip_init: bool = False,
        enable_final_token: bool = True,
    ):
        """
        Set up thought tokens for query representation.

        Args:
            num_thought_tokens: Number of thought tokens (K)
            semantic_init_token: Token to use for semantic initialization (optional)
            skip_init: If True, skip random initialization (for loading checkpoints)
            enable_final_token: Whether to add/use <final> token.
        """
        if num_thought_tokens <= 0 and not enable_final_token:
            self._num_thought_tokens = 0
            self.mbeir_num_thought_tokens = 0
            self._thought_token_ids = []
            self._final_token_id = None
            self.enable_final_token = False
            return

        self._num_thought_tokens = max(0, num_thought_tokens)
        self.mbeir_num_thought_tokens = max(0, num_thought_tokens)
        self.enable_final_token = bool(enable_final_token)

        # Generate token strings
        thought_tokens = get_all_thought_tokens(self._num_thought_tokens)
        all_reasoning_tokens = list(thought_tokens)
        if enable_final_token:
            all_reasoning_tokens.append(FINAL_TOKEN)

        if not all_reasoning_tokens:
            self._thought_token_ids = []
            self._final_token_id = None
            self.enable_final_token = False
            return

        # Add to tokenizer
        num_added = self.processor.tokenizer.add_special_tokens({
            'additional_special_tokens': all_reasoning_tokens
        })

        # Get token IDs
        self._final_token_id = (
            self.processor.tokenizer.convert_tokens_to_ids(FINAL_TOKEN)
            if enable_final_token
            else None
        )
        self._thought_token_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(t) for t in thought_tokens
        ]

        # Verify uniqueness
        all_ids = list(self._thought_token_ids)
        if self._final_token_id is not None:
            all_ids.append(self._final_token_id)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError(f"Reasoning token IDs must be unique, got IDs={all_ids}")

        # Resize embeddings if needed
        current_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(self.processor.tokenizer)

        if new_vocab_size > current_vocab_size:
            self.model.resize_token_embeddings(new_vocab_size)

            # if not skip_init:
            #     # Random initialization for newly added tokens
            #     newly_added_ids = [tid for tid in all_ids if tid >= current_vocab_size]
            #     self._random_init_reasoning_tokens(newly_added_ids)
            if not skip_init:
                # Semantic initialization for newly added tokens
                newly_added_ids = [tid for tid in all_ids if tid >= current_vocab_size]
                self._random_init_reasoning_tokens(
                    newly_added_ids, 
                    semantic_token=semantic_init_token  # 传入句号 "."
                )

        # Wrap the embedding layer to make only thought/final tokens trainable.
        # This avoids allocating a huge gradient tensor for the entire vocab.
        base_embed = self.model.get_input_embeddings()
        wrapped_embed = ReasoningTokenEmbeddingWrapper(base_embed, all_ids)
        self.model.set_input_embeddings(wrapped_embed)

        print(f"✓ Setup {self._num_thought_tokens} thought tokens")
        print(f"  Thought token IDs: {self._thought_token_ids}")
        print(f"  Final token enabled: {enable_final_token}, id: {self._final_token_id}")
        print(f"  Wrapped embedding layer to train only {len(all_ids)} special tokens")

    # def _random_init_reasoning_tokens(self, token_ids: List[int]):
    #     """Randomly initialize embeddings for reasoning tokens."""
    #     embed_layer = self.model.get_input_embeddings()
    #     embed_weight = embed_layer.weight

    #     with torch.no_grad():
    #         for token_id in token_ids:
    #             # Initialize with normal distribution matching existing embeddings
    #             nn.init.normal_(
    #                 embed_weight[token_id],
    #                 mean=0.0,
    #                 std=embed_weight.std().item()
    #             )
    def _random_init_reasoning_tokens(self, token_ids: List[int], semantic_token: str = "."):
        """
        使用语义 Token (如句号) 进行初始化，并加入微小噪声打破对称性。
        避免纯随机初始化带来的狂暴梯度炸毁 LoRA。
        """
        embed_layer = self.model.get_input_embeddings()
        embed_weight = embed_layer.weight
        
        # 【稳妥做法】使用 encode 获取语义 token 的真实 ID
        # 使用 [-1] 是为了防止某些 tokenizer 自动在前面加空位符或 bos
        encoded_ids = self.processor.tokenizer.encode(semantic_token, add_special_tokens=False)
        if not encoded_ids:
            raise ValueError(f"无法对 semantic_token '{semantic_token}' 进行编码，请检查 tokenizer。")
        init_token_id = encoded_ids[-1]
        
        with torch.no_grad():
            # 提取 "." 的原始预训练向量作为基底 (必须 clone，防止原地修改)
            base_embed = embed_weight[init_token_id].clone()
            
            for token_id in token_ids:
                # 核心机制：以句号向量为基底，加上极小的噪声扰动 (1e-5)
                # 既保持了安全的流形位置，又让不同的 thought token 有差异化的优化起点
                noise = torch.randn_like(base_embed) * 1e-5
                embed_weight[token_id].copy_(base_embed + noise)

    def _append_thought_tokens_to_text(
        self,
        text: str,
        num_thought_tokens: int,
        use_final_token: Optional[bool] = None,
    ) -> str:
        """Append thought tokens to text string."""
        if use_final_token is None:
            include_final = self._final_token_id is not None and bool(getattr(self, "enable_final_token", False))
        else:
            include_final = self._final_token_id is not None and bool(use_final_token)
        if num_thought_tokens <= 0 and not include_final:
            return text

        thought_tokens = get_all_thought_tokens(max(0, num_thought_tokens))
        if include_final:
            thought_tokens = thought_tokens + [FINAL_TOKEN]
        if not thought_tokens:
            return text
        thought_str = " ".join(thought_tokens)
        text = text or ""
        if text.strip():
            return f"{text} {thought_str}"
        return thought_str

    def _truncate_text_only_preserve_reasoning_tokens(
        self,
        text: str,
        has_image: bool,
        num_thought_tokens: int,
        use_final_token: bool,
        max_token_length: int,
        return_debug: bool = False,
    ) -> Any:
        """
        Truncate only the raw text portion, while reserving token budget for
        reasoning tokens (<thought_i>, <final>) so they are not cut off.
        """
        tokenizer = self.processor.tokenizer

        # Build reasoning suffix and reserve its token budget.
        reasoning_tokens = get_all_thought_tokens(max(0, num_thought_tokens))
        if use_final_token and self._final_token_id is not None:
            reasoning_tokens = reasoning_tokens + [FINAL_TOKEN]
        reasoning_suffix = " ".join(reasoning_tokens)

        reserve_reasoning_len = 0
        if reasoning_suffix:
            reserve_reasoning_ids = tokenizer.encode(reasoning_suffix, add_special_tokens=False)
            reserve_reasoning_len = len(reserve_reasoning_ids)

        # Estimate non-text overhead of chat template (+ image placeholder if any).
        content = []
        if has_image:
            content.append({
                'type': 'image',
                'image': Image.new('RGB', (32, 32), color=(0, 0, 0)),
                'min_pixels': 4 * 32 * 32,
                'max_pixels': 1800 * 32 * 32,
            })
        content.append({'type': 'text', 'text': ''})
        overhead_conversation = [[
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content},
        ]]
        overhead_prompt = self.processor.apply_chat_template(
            overhead_conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        overhead_ids = tokenizer(
            overhead_prompt,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        ).get("input_ids", [[]])[0]
        overhead_len = len(overhead_ids)

        # Available token budget for the raw text only.
        available_text_len = max(0, max_token_length - overhead_len - reserve_reasoning_len)

        text_ids = tokenizer.encode(text or "", add_special_tokens=False)
        if len(text_ids) <= available_text_len:
            truncated_text = text or ""
        else:
            # Truncate from the tail of raw text only.
            text_ids = text_ids[:available_text_len]
            truncated_text = tokenizer.decode(
                text_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        if reasoning_suffix:
            if truncated_text.strip():
                final_text = f"{truncated_text} {reasoning_suffix}"
            else:
                final_text = reasoning_suffix
        else:
            final_text = truncated_text

        if return_debug:
            return final_text, {
                "raw_text_token_len": len(tokenizer.encode(text or "", add_special_tokens=False)),
                "available_text_len": int(available_text_len),
                "truncated_text_token_len": len(tokenizer.encode(truncated_text or "", add_special_tokens=False)),
                "reserve_reasoning_len": int(reserve_reasoning_len),
                "overhead_len": int(overhead_len),
                "final_text_token_len": len(tokenizer.encode(final_text or "", add_special_tokens=False)),
                "was_truncated": bool(len(tokenizer.encode(text or "", add_special_tokens=False)) > available_text_len),
            }

        return final_text

    @torch.no_grad()
    def estimate_max_prompt_token_length(self, texts: List[str], num_thought_tokens: int = 0) -> int:
        """Estimate max token length after chat template (no truncation).

        Used to *drop* overlong samples before they ever reach the model forward,
        avoiding OOM and keeping DDP steps consistent.

        Note: this is a text-only estimate (images omitted). In this project the
        dataloader already resizes images to a fixed size, so text length is the
        dominant source of long sequences.
        """
        if not texts:
            return 0

        if num_thought_tokens > 0:
            texts = [self._append_thought_tokens_to_text(t, num_thought_tokens) for t in texts]

        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": [{"type": "text", "text": t}]},
            ]
            for t in texts
        ]

        text_prompts = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        enc = self.processor.tokenizer(
            text_prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        input_ids = enc.get("input_ids", [])
        if not input_ids:
            return 0
        return max(len(ids) for ids in input_ids)

    def _build_reasoning_attention_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        num_thought_tokens: int,
    ) -> torch.Tensor:
        """
        Build 4D causal mask where <final> token can only attend to thought tokens and itself.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1=valid, 0=padding)
            num_thought_tokens: Number of thought tokens

        Returns:
            4D attention mask [batch_size, 1, seq_len, seq_len] (0=allow, -inf=mask)
        """
        if (
            num_thought_tokens <= 0
            or self._thought_token_ids is None
            or self._final_token_id is None
            or attention_mask is None
        ):
            return attention_mask

        if attention_mask.dim() == 4:
            return attention_mask

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        #mask_dtype = torch.float32
        target_dtype = self.model.get_input_embeddings().weight.dtype
        mask_dtype = target_dtype
        min_value = torch.finfo(mask_dtype).min

        # Build standard causal mask (0=allow, -inf=mask)
        base = torch.full((seq_len, seq_len), fill_value=min_value, device=device, dtype=mask_dtype)
        base = base.masked_fill(
            torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)),
            0.0
        )
        causal = base.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()

        # Apply padding mask
        valid = attention_mask.to(device=device).bool()
        causal = causal.masked_fill(~valid[:, None, None, :], min_value)  # Mask padded keys
        causal = causal.masked_fill(~valid[:, None, :, None], min_value)  # Mask padded queries

        # Get thought token IDs tensor
        thought_ids = torch.tensor(
            self._thought_token_ids[:num_thought_tokens],
            device=device,
            dtype=input_ids.dtype,
        )

        # For each sample, restrict <final> token's attention
        for b in range(batch_size):
            final_positions = (input_ids[b] == self._final_token_id).nonzero(as_tuple=True)[0]
            if final_positions.numel() == 0:
                continue
            final_pos = final_positions[-1].item()

            # Identify thought and final tokens
            is_thought = (input_ids[b].unsqueeze(0) == thought_ids.unsqueeze(1)).any(dim=0)
            is_final = input_ids[b] == self._final_token_id
            allowed_keys = (is_thought | is_final) & valid[b]

            # Mask everything for final token, then unmask allowed keys
            causal[b, 0, final_pos, :] = min_value
            causal[b, 0, final_pos, allowed_keys] = 0.0

        return causal

    def _get_single_vector_embedding(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.LongTensor,
        num_thought_tokens: int,
        use_final_token: bool = False,
    ) -> torch.Tensor:
        """
        Extract single-vector embedding.

        Cases:
        1. num_thought_tokens > 0 and enable_final_token: use <final> token embedding
        2. enable_final_token (but num_thought_tokens=0): use <final> token embedding
        3. Otherwise: use last valid token embedding (standard pooling)
        
        This method correctly handles different token configurations:
        - When thought tokens are present, the model should have appended <final> at the end
        - When only final token is enabled, it should still be present even if num_thought_tokens=0
        - The attention_mask ensures we only look at valid (non-padded) positions
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Check if we should use final token for pooling
        # Final token pooling is explicitly controlled by caller so query/candidate
        # behavior can be strictly separated.
        use_final_token = bool(use_final_token) and self._final_token_id is not None
        if use_final_token:
            # Find <final> token position - should be present in all samples
            # (either because num_thought_tokens > 0 OR enable_final_token=True)
            final_mask = (input_ids == self._final_token_id)
            
            # Check if <final> token exists - should be true for all samples
            has_final = final_mask.any(dim=1)
            if not has_final.all():
                # Log diagnostic information for debugging
                missing_indices = (~has_final).nonzero(as_tuple=True)[0]
                raise ValueError(
                    f"<final> token (ID={self._final_token_id}) not found in {missing_indices.numel()} "
                    f"sample(s) out of {batch_size}. "
                    f"num_thought_tokens={num_thought_tokens}, enable_final_token={use_final_token}. "
                    f"This indicates a mismatch between training and evaluation token configuration."
                )
            
            # Get position of <final> token (use argmax to get first occurrence)
            positions = final_mask.long().argmax(dim=1)

            # Extract embeddings at these positions
            batch_indices = torch.arange(batch_size, device=device)
            pooled_output = hidden_states[batch_indices, positions]
            
            if device != hidden_states.device:
                pooled_output = pooled_output.to(device)
        else:
            # Standard last valid token pooling
            # Find the position of the last valid (non-padded) token for each sample
            
            # attention_mask: 1 = valid, 0 = padding
            # We need to find the last position where attention_mask == 1
            
            # Ensure attention_mask is on the correct device and dtype
            mask = attention_mask.bool().to(device)
            
            if mask.dim() == 4:
                # If 4D mask was passed, extract the 2D component
                # This handles case where _build_reasoning_attention_mask returned 4D mask
                mask = mask[:, 0, :, -1]  # Take last query dimension
            
            # Flip the mask to find the last True value from the end
            flipped_mask = mask.flip(dims=[1])
            
            # Find first True in flipped mask (= last True in original)
            last_positions = flipped_mask.long().argmax(dim=1)
            
            # Convert flipped position back to original position
            # If all positions are 0 (no valid tokens), this will be seq_len-1
            seq_len = mask.shape[1]
            col = seq_len - last_positions - 1
            
            # Handle edge case: ensure we don't go out of bounds
            col = torch.clamp(col, min=0, max=seq_len - 1)
            
            row = torch.arange(batch_size, device=device)
            pooled_output = hidden_states[row, col]

        return F.normalize(pooled_output, dim=-1)

    def _compute_position_ids_for_recursive_step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """Compute Qwen3-VL RoPE position ids for recursive inputs_embeds forward."""
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()
        else:
            attention_mask_tensor = attention_mask

        position_ids, _ = self.model.model.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=attention_mask_tensor,
        )
        return position_ids

    def _forward_recursive_embedding_reasoning(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        num_reasoning_steps: int,
        **kwargs,
    ) -> Qwen3ThoughtOutput:
        """Recursive embedding-token reasoning:
        Step 0 gets e_0 from original input.
        Step k appends e_{k-1} as one embedding token and produces e_k.
        """
        # Step 0
        base_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        e_prev = self._get_single_vector_embedding(
            hidden_states=base_outputs.last_hidden_state,
            attention_mask=attention_mask,
            input_ids=input_ids,
            num_thought_tokens=0,
            use_final_token=False,
        )
        step_embeddings: List[torch.Tensor] = [e_prev]

        if num_reasoning_steps <= 0:
            return Qwen3ThoughtOutput(
                single_vec_emb=e_prev,
                thought_embeddings=step_embeddings,
            )

        batch_size = input_ids.size(0)
        device = input_ids.device
        embed_layer = self.model.get_input_embeddings()
        append_token_id = self.processor.tokenizer.pad_token_id
        if append_token_id is None:
            append_token_id = self.processor.tokenizer.eos_token_id
        if append_token_id is None:
            append_token_id = 0

        running_input_ids = input_ids
        running_attention_mask = attention_mask
        reasoning_history: List[torch.Tensor] = [e_prev]

        if bool(getattr(self, "debug_mode", False)):
            self._debug_log(
                "recursive-start: "
                f"num_reasoning_steps={num_reasoning_steps}, "
                f"batch_size={batch_size}, "
                f"base_seq_len={input_ids.shape[1]}, "
                f"e0_norm_mean={float(e_prev.norm(dim=-1).mean().item()):.6f}"
            )

        # Remove keys that will be supplied explicitly for recursive passes.
        forward_kwargs = dict(kwargs)
        forward_kwargs.pop("position_ids", None)
        forward_kwargs.pop("input_ids", None)
        forward_kwargs.pop("inputs_embeds", None)
        forward_kwargs.pop("attention_mask", None)

        for step_idx in range(1, num_reasoning_steps + 1):
            append_ids = torch.full(
                (batch_size, 1),
                fill_value=append_token_id,
                dtype=running_input_ids.dtype,
                device=device,
            )
            append_mask = torch.ones(
                (batch_size, 1),
                dtype=running_attention_mask.dtype,
                device=device,
            )

            running_input_ids = torch.cat([running_input_ids, append_ids], dim=1)
            running_attention_mask = torch.cat([running_attention_mask, append_mask], dim=1)

            step_inputs_embeds = embed_layer(running_input_ids)
            step_inputs_embeds = step_inputs_embeds.clone()
            history_tensor = torch.stack(reasoning_history, dim=1).to(step_inputs_embeds.dtype)
            step_inputs_embeds[:, -history_tensor.size(1):, :] = history_tensor

            if bool(getattr(self, "debug_mode", False)):
                valid_len0 = int(running_attention_mask[0].sum().item()) if running_attention_mask.ndim == 2 else -1
                self._debug_log(
                    "recursive-step-input: "
                    f"step={step_idx}/{num_reasoning_steps}, "
                    f"seq_len={running_input_ids.shape[1]}, "
                    f"valid_len_sample0={valid_len0}, "
                    f"history_len={history_tensor.shape[1]}, "
                    f"tail_replaced={history_tensor.shape[1]}"
                )

            position_ids = self._compute_position_ids_for_recursive_step(
                input_ids=running_input_ids,
                attention_mask=running_attention_mask,
                image_grid_thw=forward_kwargs.get("image_grid_thw", None),
                video_grid_thw=forward_kwargs.get("video_grid_thw", None),
            )

            step_outputs = self.model(
                input_ids=None,
                inputs_embeds=step_inputs_embeds,
                attention_mask=running_attention_mask,
                position_ids=position_ids,
                **forward_kwargs,
            )

            e_prev = self._get_single_vector_embedding(
                hidden_states=step_outputs.last_hidden_state,
                attention_mask=running_attention_mask,
                input_ids=running_input_ids,
                num_thought_tokens=0,
                use_final_token=False,
            )
            step_embeddings.append(e_prev)
            reasoning_history.append(e_prev)

            if bool(getattr(self, "debug_mode", False)):
                self._debug_log(
                    "recursive-step-output: "
                    f"step={step_idx}/{num_reasoning_steps}, "
                    f"ek_norm_mean={float(e_prev.norm(dim=-1).mean().item()):.6f}, "
                    f"stored_steps={len(step_embeddings)}"
                )

        final_embedding = F.normalize(step_embeddings[-1], dim=-1)
        return Qwen3ThoughtOutput(
            single_vec_emb=final_embedding,
            thought_embeddings=step_embeddings,
        )

    def _extract_reasoning_step_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        num_thought_tokens: int,
        use_final_token: bool,
    ) -> Optional[List[torch.Tensor]]:
        """
        Extract normalized embeddings for each reasoning step:
        [thought_1, thought_2, ..., thought_{K-1}, final].

        Returns None when no reasoning tokens are requested.
        """
        step_ids: List[int] = []

        if num_thought_tokens > 0 and self._thought_token_ids is not None:
            step_ids.extend(self._thought_token_ids[:num_thought_tokens])

        if bool(use_final_token) and self._final_token_id is not None:
            step_ids.append(self._final_token_id)

        if len(step_ids) == 0:
            return None

        batch_size = hidden_states.size(0)
        device = hidden_states.device
        batch_indices = torch.arange(batch_size, device=device)
        step_embeddings: List[torch.Tensor] = []

        for token_id in step_ids:
            token_mask = (input_ids == token_id)
            has_token = token_mask.any(dim=1)
            if not has_token.all():
                missing_indices = (~has_token).nonzero(as_tuple=True)[0]
                raise ValueError(
                    f"Reasoning token (ID={token_id}) not found in "
                    f"{missing_indices.numel()} sample(s) out of {batch_size}."
                )

            positions = token_mask.long().argmax(dim=1)
            emb = hidden_states[batch_indices, positions]
            step_embeddings.append(F.normalize(emb, dim=-1))

        return step_embeddings

    def _preprocess_inputs(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Image.Image]] = None,
        num_thought_tokens: int = 0,
        use_final_token: Optional[bool] = None,
        max_token_length=None,   # Change default from 2048 to None
        max_visual_pixels=None   # Change default from 501760 to None
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess inputs into model-ready format.

        Args:
            texts: List of text strings
            images: List of PIL images
            num_thought_tokens: Number of thought tokens to append

        Returns:
            Dict with input_ids, attention_mask, pixel_values, etc.
        """
        safe_max_token_length = max_token_length or getattr(self, "max_token_length", 1500)
        safe_max_visual_pixels = max_visual_pixels or getattr(self, "max_visual_pixels", 501760)
        effective_use_final_token = (
            bool(getattr(self, "enable_final_token", False))
            if use_final_token is None
            else bool(use_final_token)
        )
        self._debug_preprocess_calls += 1
        debug_now = bool(getattr(self, "debug_mode", False)) and (
            self._debug_preprocess_calls % max(1, int(getattr(self, "debug_preprocess_print_freq", 1))) == 0
        )
        debug_sample_info = None
        original_texts = list(texts) if texts is not None else None

        # Append static thought/final tokens only in legacy mode.
        legacy_static_reasoning = (
            (not bool(getattr(self, "use_recursive_embedding_reasoning", True)))
            and (num_thought_tokens > 0 or effective_use_final_token)
        )
        use_special_tokens = legacy_static_reasoning
        if texts:
            if use_special_tokens:
                processed_texts = []
                for i, t in enumerate(texts):
                    has_image = bool(images and i < len(images))
                    processed = self._truncate_text_only_preserve_reasoning_tokens(
                        text=t,
                        has_image=has_image,
                        num_thought_tokens=num_thought_tokens,
                        use_final_token=effective_use_final_token,
                        max_token_length=safe_max_token_length,
                        return_debug=(debug_now and i == 0),
                    )
                    if debug_now and i == 0:
                        processed_text, meta = processed
                        debug_sample_info = {
                            "raw_text": t,
                            "processed_text": processed_text,
                            "meta": meta,
                            "has_image": has_image,
                        }
                        processed_texts.append(processed_text)
                    else:
                        processed_texts.append(processed)
                texts = processed_texts
            else:
                # Non-reasoning path keeps processor-level truncation behavior.
                texts = list(texts)

        # Build conversation format
        conversations = []
        for i in range(len(texts) if texts else len(images) if images else 0):
            content = []

            # 按需求改为先拼接文本，再拼接图像。
            # 顺序为 [Text] [Image]。
            # Add image (Second!)
            if images and i < len(images):
                content.append({
                    'type': 'image',
                    'image': images[i],
                    'min_pixels': 4 * 32 * 32,
                    'max_pixels': 1800 * 32 * 32,
                })
            # Add text (First!)
            if texts and i < len(texts):
                content.append({'type': 'text', 'text': texts[i]})

            

            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": content}
            ])

        # Apply chat template
        text_prompts = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        # Full payload debug before model encoding.
        if debug_now:
            self._debug_log(
                "full-pre-encoding-summary: "
                f"batch_size={len(conversations)}, "
                f"use_special_tokens={use_special_tokens}, "
                f"num_thought_tokens={num_thought_tokens}, "
                f"use_final_token={effective_use_final_token}, "
                f"max_token_length={safe_max_token_length}, "
                f"max_visual_pixels={safe_max_visual_pixels}"
            )

            def _serialize_conversation(conv):
                serialized = []
                for turn in conv:
                    turn_copy = {"role": turn.get("role"), "content": []}
                    for c in turn.get("content", []):
                        if c.get("type") == "image":
                            img_obj = c.get("image", None)
                            if isinstance(img_obj, Image.Image):
                                img_desc = f"<PIL.Image mode={img_obj.mode} size={img_obj.size}>"
                            else:
                                img_desc = str(type(img_obj))
                            turn_copy["content"].append(
                                {
                                    "type": "image",
                                    "image": img_desc,
                                    "min_pixels": c.get("min_pixels"),
                                    "max_pixels": c.get("max_pixels"),
                                }
                            )
                        else:
                            turn_copy["content"].append(c)
                    serialized.append(turn_copy)
                return serialized

            total_items = len(conversations)
            for idx in range(total_items):
                raw_input_text_full = ""
                if original_texts and idx < len(original_texts):
                    raw_input_text_full = str(original_texts[idx])
                model_input_text_full = ""
                if texts and idx < len(texts):
                    model_input_text_full = str(texts[idx])
                has_image = bool(images and idx < len(images) and images[idx] is not None)
                self._debug_log(f"full-item[{idx}]-has_image={has_image}")
                self._debug_log(f"full-item[{idx}]-raw-input-text='{raw_input_text_full}'")
                self._debug_log(f"full-item[{idx}]-model-input-text='{model_input_text_full}'")
                if idx < len(conversations):
                    self._debug_log(
                        f"full-item[{idx}]-conversation={_serialize_conversation(conversations[idx])}"
                    )
                if idx < len(text_prompts):
                    self._debug_log(f"full-item[{idx}]-prompt='{text_prompts[idx]}'")

        # Tokenization-stage debug (after tokenization, before model encoding)
        if debug_now and text_prompts:
            tok = self.processor.tokenizer
            preview_k = int(getattr(self, "debug_token_preview_len", 12))
            sample_prompt = text_prompts[0]

            # Untruncated tokenization check on text prompt
            prompt_ids_full = tok.encode(sample_prompt, add_special_tokens=False)

            # Simulated truncation on text prompt (for visibility only)
            trunc_side_backup_local = getattr(tok, "truncation_side", "right")
            tok.truncation_side = "left" if use_special_tokens else "right"
            try:
                prompt_ids_trunc = tok(
                    [sample_prompt],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=safe_max_token_length,
                    padding=False,
                    return_attention_mask=False,
                )["input_ids"][0]
            finally:
                tok.truncation_side = trunc_side_backup_local

            head_ids = prompt_ids_trunc[:preview_k]
            tail_ids = prompt_ids_trunc[-preview_k:] if len(prompt_ids_trunc) > preview_k else prompt_ids_trunc
            head_tokens = tok.convert_ids_to_tokens(head_ids)
            tail_tokens = tok.convert_ids_to_tokens(tail_ids)

            self._debug_log(
                "tokenize-check: "
                f"prompt_tokens_full={len(prompt_ids_full)}, "
                f"prompt_tokens_after_text_trunc={len(prompt_ids_trunc)}, "
                f"max_token_length={safe_max_token_length}, "
                f"truncation_side={'left' if use_special_tokens else 'right'}"
            )
            self._debug_log(f"tokenize-head-ids={head_ids}")
            self._debug_log(f"tokenize-head-tokens={head_tokens}")
            self._debug_log(f"tokenize-tail-ids={tail_ids}")
            self._debug_log(f"tokenize-tail-tokens={tail_tokens}")

            # Special token position check on tokenized text prompt
            expected_debug_ids = []
            if use_special_tokens:
                if num_thought_tokens > 0 and self._thought_token_ids is not None:
                    expected_debug_ids.extend(self._thought_token_ids[:num_thought_tokens])
                if effective_use_final_token and self._final_token_id is not None:
                    expected_debug_ids.append(self._final_token_id)

            if expected_debug_ids:
                pos_info = {}
                for sid in expected_debug_ids:
                    pos_info[sid] = [idx for idx, tid in enumerate(prompt_ids_trunc) if tid == sid]
                self._debug_log(f"tokenize-special-token-positions={pos_info}")

        # Process vision info
        try:
            images_processed, video_inputs, video_kwargs = process_vision_info(
                conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
        except:
            images_processed = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None
        # ================== [核心修改：Processor 截断] ==================
        # 注意：必须在 processor(...) 上启用 truncation，max_length 才会生效。
        # 同时我们把 truncation_side 设为 "left" 或 "right"：
        # - 对于需要保留 thought/final tokens 的序列（query），使用左截断
        # - 对于标准 embedding（doc），使用右截断
        #
        # 具体规则：
        # 1. num_thought_tokens > 0：左截断（保留末尾的 thought 和 final tokens）
        # 2. enable_final_token=True 且 num_thought_tokens=0：左截断（保留末尾的 final token）
        # 3. 都不满足：右截断（保留文章开头的核心信息）
        trunc_side_backup = getattr(self.processor.tokenizer, "truncation_side", None)
        
        if use_special_tokens:
            self.processor.tokenizer.truncation_side = "left"
        else:
            self.processor.tokenizer.truncation_side = "right"
        
        try:
            inputs = self.processor(
                text=text_prompts,
                images=images_processed,
                truncation=True,
                max_length=safe_max_token_length, # Use safe variable
                padding=True,
                do_resize=False,
                return_tensors='pt',
                max_pixels=safe_max_visual_pixels, # Use safe variable
            )

            if debug_now:
                shapes = {k: tuple(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                self._debug_log(f"full-processor-input-shapes={shapes}")
                if "input_ids" in inputs:
                    for idx in range(inputs["input_ids"].shape[0]):
                        self._debug_log(
                            f"full-item[{idx}]-input_ids={inputs['input_ids'][idx].tolist()}"
                        )
                if "attention_mask" in inputs:
                    for idx in range(inputs["attention_mask"].shape[0]):
                        self._debug_log(
                            f"full-item[{idx}]-attention_mask={inputs['attention_mask'][idx].tolist()}"
                        )

            if debug_now and debug_sample_info is not None and "attention_mask" in inputs:
                max_chars = int(getattr(self, "debug_max_text_chars", 120))
                raw_preview = str(debug_sample_info["raw_text"] or "")
                proc_preview = str(debug_sample_info["processed_text"] or "")
                if len(raw_preview) > max_chars:
                    raw_preview = raw_preview[:max_chars] + "..."
                if len(proc_preview) > max_chars:
                    proc_preview = proc_preview[:max_chars] + "..."

                seq_len = int(inputs["attention_mask"][0].sum().item())
                meta = debug_sample_info["meta"]
                self._debug_log(
                    "truncate-check: "
                    f"has_image={debug_sample_info['has_image']}, "
                    f"raw_text_tokens={meta['raw_text_token_len']}, "
                    f"available_text_tokens={meta['available_text_len']}, "
                    f"truncated_text_tokens={meta['truncated_text_token_len']}, "
                    f"reserve_reasoning_tokens={meta['reserve_reasoning_len']}, "
                    f"overhead_tokens={meta['overhead_len']}, "
                    f"final_text_tokens={meta['final_text_token_len']}, "
                    f"final_seq_len_after_processor={seq_len}, "
                    f"was_truncated={meta['was_truncated']}"
                )
                self._debug_log(f"raw_text_preview='{raw_preview}'")
                self._debug_log(f"processed_text_preview='{proc_preview}'")

            # Strict guarantee: reasoning special tokens must exist in every sample.
            if use_special_tokens and "input_ids" in inputs:
                expected_ids = []
                if num_thought_tokens > 0 and self._thought_token_ids is not None:
                    expected_ids.extend(self._thought_token_ids[:num_thought_tokens])
                if effective_use_final_token and self._final_token_id is not None:
                    expected_ids.append(self._final_token_id)

                input_ids = inputs["input_ids"]
                for token_id in expected_ids:
                    has_token = (input_ids == token_id).any(dim=1)
                    if not has_token.all():
                        missing_count = int((~has_token).sum().item())
                        raise ValueError(
                            f"Special token {token_id} was truncated or missing in "
                            f"{missing_count} sample(s). Check max_token_length={safe_max_token_length}."
                        )
        finally:
            if trunc_side_backup is not None:
                self.processor.tokenizer.truncation_side = trunc_side_backup

        return inputs

    def _forward_embeddings(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        num_thought_tokens: int = 0,
        use_final_token: bool = False,
        **kwargs
    ) -> Qwen3ThoughtOutput:
        """
        Forward pass with thought token support.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            num_thought_tokens: Number of thought tokens
            **kwargs: Additional args (pixel_values, image_grid_thw, etc.)

        Returns:
            Qwen3ThoughtOutput with single_vec_emb
        """
        # Default path: recursive embedding reasoning (replaces static thought token concat).
        if bool(getattr(self, "use_recursive_embedding_reasoning", True)):
            return self._forward_recursive_embedding_reasoning(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_reasoning_steps=max(0, int(num_thought_tokens)),
                **kwargs,
            )

        # Legacy fallback: single pass + static special-token extraction.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        single_vec_emb = self._get_single_vector_embedding(
            hidden_states=outputs.last_hidden_state,
            attention_mask=attention_mask,
            input_ids=input_ids,
            num_thought_tokens=num_thought_tokens,
            use_final_token=use_final_token,
        )

        thought_embeddings = self._extract_reasoning_step_embeddings(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            num_thought_tokens=num_thought_tokens,
            use_final_token=use_final_token,
        )

        return Qwen3ThoughtOutput(single_vec_emb=single_vec_emb, thought_embeddings=thought_embeddings)

    # ------------------------------------------------------------------
    # MBEIR Interface
    # ------------------------------------------------------------------

    def get_img_preprocess_fn(self):
        """Returns image preprocessing function for MBEIR data loader."""
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
        """Returns a passthrough function that preserves raw text strings."""
        def passthrough_wrapper(texts: List[str]) -> RawTextBatch:
            return RawTextBatch(texts)
        return passthrough_wrapper

    def encode_mbeir_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode MBEIR batch with configurable query/doc encoding.

        Default behavior is asymmetric (query uses thought/final, candidate doesn't).
        If `self.mbeir_symmetric_encoding=True`, both query and candidate use the
        same thought/final-token settings.
        
        Debugging info:
        - Logs token configuration for each batch type
        - Verifies that <final> token is present when expected
        """
        is_query_batch = "qid_list" in batch and batch["qid_list"] is not None
        is_candidate_batch = "did_list" in batch and batch["did_list"] is not None
        id_list = batch.get("did_list") or batch.get("qid_list")

        if id_list is None:
            raise ValueError("id_list (did_list or qid_list) not found in batch.")

        symmetric_mode = bool(
            getattr(self, 'mbeir_symmetric_encoding', False)
            or getattr(self, 'symmetric_query_candidate_encoding', False)
        )

        if symmetric_mode:
            num_thought_tokens = getattr(self, 'mbeir_num_thought_tokens', 0)
            use_final_token = bool(getattr(self, 'enable_final_token', False))
            if is_query_batch and not is_candidate_batch:
                batch_type = "QUERY(SYMMETRIC)"
            elif is_candidate_batch and not is_query_batch:
                batch_type = "CANDIDATE(SYMMETRIC)"
            else:
                batch_type = "MIXED(SYMMETRIC)"
        else:
            # Asymmetric encoding: only queries use thought/final tokens
            if is_query_batch and not is_candidate_batch:
                num_thought_tokens = getattr(self, 'mbeir_num_thought_tokens', 0)
                use_final_token = bool(getattr(self, 'enable_final_token', False))
                batch_type = "QUERY"
            else:
                num_thought_tokens = 0
                use_final_token = False
                batch_type = "CANDIDATE" if is_candidate_batch else "MIXED"

        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask = batch["txt_mask_batched"]
        image_mask = batch["image_mask_batched"]

        batch_size = image_batched.size(0)
        device = self.model.device

        # Log token configuration
        import sys
        # print(
        #     f"[encode_mbeir_batch] Type={batch_type}, "
        #     f"Size={batch_size}, TT={num_thought_tokens}, "
        #     f"FT={getattr(self, 'enable_final_token', False)}, "
        #     f"FinalTokenID={self._final_token_id}",
        #     file=sys.stderr
        # )

        # Group by modality
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

        # Get embedding dimension
        embed_dim = self.model.config.text_config.hidden_size
        embeddings = torch.zeros(batch_size, embed_dim, device=device, dtype=self.model.dtype)

        # Encode each modality group
        if text_only_idx:
            texts = txt_batched[text_only_idx]
            inputs = self._preprocess_inputs(
                texts=texts,
                num_thought_tokens=num_thought_tokens,
                use_final_token=use_final_token,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = self._forward_embeddings(
                num_thought_tokens=num_thought_tokens,
                use_final_token=use_final_token,
                **inputs,
            )
            embeddings[text_only_idx] = output.single_vec_emb.to(embeddings.dtype)

        if image_only_idx:
            pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in image_only_idx]
            inputs = self._preprocess_inputs(
                texts=[""] * len(pil_images),
                images=pil_images,
                num_thought_tokens=num_thought_tokens,
                use_final_token=use_final_token,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = self._forward_embeddings(
                num_thought_tokens=num_thought_tokens,
                use_final_token=use_final_token,
                **inputs,
            )
            embeddings[image_only_idx] = output.single_vec_emb.to(embeddings.dtype)

        if multimodal_idx:
            texts = txt_batched[multimodal_idx]
            pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in multimodal_idx]
            inputs = self._preprocess_inputs(
                texts=texts,
                images=pil_images,
                num_thought_tokens=num_thought_tokens,
                use_final_token=use_final_token,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = self._forward_embeddings(
                num_thought_tokens=num_thought_tokens,
                use_final_token=use_final_token,
                **inputs,
            )
            embeddings[multimodal_idx] = output.single_vec_emb.to(embeddings.dtype)

        return embeddings, id_list

    # ------------------------------------------------------------------
    # Model lifecycle methods
    # ------------------------------------------------------------------

    # 替换掉原有的 forward 方法
    def forward(
        self,
        encode_mbeir_batch: bool = False,
        num_thought_tokens: int = 0,
        use_final_token: bool = False,
        **kwargs
    ):
        """
        统一的前向传播入口。
        DDP 训练必须通过这里才能触发梯度同步 Hook。
        """
        if encode_mbeir_batch:
            # 推理阶段：走现有的 MBEIR 格式化逻辑
            return self.encode_mbeir_batch(kwargs)
        else:
            # 训练阶段：直接走底层带 4D Mask 的 Embedding 逻辑
            return self._forward_embeddings(
                num_thought_tokens=num_thought_tokens,
                use_final_token=use_final_token,
                **kwargs,
            )

    def __call__(self, encode_mbeir_batch: bool = False, **kwargs):
        return self.forward(encode_mbeir_batch=encode_mbeir_batch, **kwargs)

    

    @property
    def device(self):
        return self.model.device
