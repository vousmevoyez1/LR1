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


class JinaEmbeddingsV4Processor(Qwen2_5_VLProcessor):
    def __init__(self, *args, **kwargs) -> None:
        Qwen2_5_VLProcessor.__init__(self, *args, **kwargs)
        self.assistant_prefix_len = 58
        self.text_max_length = 32768

    def process_images(
        self,
        images: Union[List[Image.Image], List[List[Image.Image]]],
    ) -> BatchFeature:

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
                text_doc.append(template[self.assistant_prefix_len :])

        else:
            images = cast(List[Image.Image], images)
            text_doc = [
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
            ] * len(images)
            # print(text_doc[0])
            # assert 1==2

        # The following code is a hack to make sure the scatter in DDP is done correctly when training on multiple GPUs
        batch_doc = self(text=text_doc, images=images, padding="longest", return_tensors="pt")  # type: ignore
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
    ) -> BatchFeature:

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
            # print(text)
            # assert 1==2
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
    ) -> BatchFeature:
        """
        Process multimodal (image + text) inputs using Jina V4's native format.

        Creates prompts in the format:
        <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>\n

        Args:
            images: List of PIL Images
            texts: List of text strings (one per image)
            max_length: Maximum token length
            prefix: Optional prefix for text (e.g., "Query")

        Returns:
            BatchFeature with input_ids, attention_mask, pixel_values, image_grid_thw
        """
        max_length = (
            self.text_max_length
            if max_length is None
            else min(max_length, self.text_max_length)
        )
        overhead_tokens = len(self.tokenizer.encode(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n", 
            add_special_tokens=False
        ))
        # Construct multimodal prompts
        text_max_length = max_length - overhead_tokens - 256  # image_token_count 需动态计算
    
    # 先截断文本
        text_prompts = []
        for text in texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) > text_max_length:
                encoded = encoded[:text_max_length]
                text = self.tokenizer.decode(encoded)
            prompt = f"<|im_start|>user\n{text}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
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
        single_vec_emb (torch.Tensor, optional): Single-vector embeddings.
        multi_vec_emb (torch.Tensor, optional): Multi-vector embeddings.
        reasoning_trajectory (List[torch.Tensor], optional): List of pooled embeddings from each reasoning step.
    """

    vlm_last_hidden_states: Optional[torch.Tensor] = None
    single_vec_emb: Optional[torch.Tensor] = None
    multi_vec_emb: Optional[torch.Tensor] = None
    reasoning_trajectory: Optional[List[torch.Tensor]] = None


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

        # Debug switches for inspecting exact encoded model inputs.
        # Enable via:
        #   export JINA_V4_DEBUG_INPUT=0
        # Optional:
        #   export JINA_V4_DEBUG_INPUT_MAX_ROWS=8
        # Backward-compatible aliases from v4o path are also supported.
        self.debug_input_dump = bool(
            int(
                os.environ.get(
                    "JINA_V4_DEBUG_INPUT",
                    os.environ.get("JINA_V4O_DEBUG_INPUT", "0"),
                )
            )
        )
        self.debug_input_max_rows = int(
            os.environ.get(
                "JINA_V4_DEBUG_INPUT_MAX_ROWS",
                os.environ.get("JINA_V4O_DEBUG_INPUT_MAX_ROWS", "4"),
            )
        )

    def _debug_dump_encoding_inputs(
        self,
        *,
        source: str,
        task_label: Union[str, List[str], None],
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        reason_steps: int = 0,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print detailed model-input debug info right before encoding forward."""
        if not getattr(self, "debug_input_dump", False):
            return

        print("\n" + "=" * 88)
        print("[JinaV4 DEBUG] Model encoding input dump")
        print(f"[JinaV4 DEBUG] source={source}")
        print(f"[JinaV4 DEBUG] task_label={task_label}")
        print(f"[JinaV4 DEBUG] reason_steps={reason_steps}")

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

    def _debug_dump_mbeir_batch_inputs(
        self,
        *,
        batch: Dict[str, Any],
        text_only_idx: List[int],
        image_only_idx: List[int],
        multimodal_idx: List[int],
    ) -> None:
        """Print raw MBEIR batch inputs before processor tokenization/encoding."""
        if not getattr(self, "debug_input_dump", False):
            return

        txt_batched = batch.get("txt_batched")
        txt_mask = batch.get("txt_mask_batched")
        image_mask = batch.get("image_mask_batched")
        qid_list = batch.get("qid_list")
        did_list = batch.get("did_list")

        print("\n" + "=" * 88)
        print("[JinaV4 DEBUG] Raw MBEIR batch input dump")
        print(f"[JinaV4 DEBUG] qid_list_present={qid_list is not None}")
        print(f"[JinaV4 DEBUG] did_list_present={did_list is not None}")
        print(
            f"[JinaV4 DEBUG] modality_groups: text_only={len(text_only_idx)}, "
            f"image_only={len(image_only_idx)}, multimodal={len(multimodal_idx)}"
        )

        if isinstance(txt_mask, torch.Tensor):
            print(f"[JinaV4 DEBUG] txt_mask={txt_mask.detach().cpu().tolist()}")
        if isinstance(image_mask, torch.Tensor):
            print(f"[JinaV4 DEBUG] image_mask={image_mask.detach().cpu().tolist()}")

        max_rows = int(getattr(self, "debug_input_max_rows", 4))
        if txt_batched is not None:
            batch_size = len(txt_batched)
            for i in range(min(max_rows, batch_size)):
                text_value = txt_batched[i]
                print(f"[JinaV4 DEBUG] row={i} raw_text={text_value}")

        print("=" * 88 + "\n")

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
    #获取最后一层隐藏状态
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

    def get_reasoning_embeddings_for_training(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        reason_steps: int = 0,
        return_trajectory: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Training-friendly multi-step reasoning WITHOUT KV cache.
        
        Unlike get_reasoning_embeddings which uses KV cache for efficiency,
        this method performs full forward passes at each step, ensuring proper
        gradient flow for training.
        
        The approach:
        1. Get initial hidden states from input
        2. For each reasoning step:
           - Pool hidden states to get a "thought vector"
           - Append it as a virtual token embedding to the input embeddings
           - Run a FULL forward pass (no cache) on extended sequence
        3. Return final embeddings
        
        Args:
            task_label: Task identifier for LoRA adapter selection
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            reason_steps: Number of reasoning iterations (0 = no reasoning)
            return_trajectory: Whether to return intermediate pooled embeddings
            **kwargs: Additional arguments (pixel_values, image_grid_thw, etc.)
            
        Returns:
            Tuple of (final_hidden_states, updated_attention_mask, reasoning_trajectory)
        """
        if reason_steps <= 0:
            # No reasoning, use original method
            hidden_states = self.get_last_hidden_states(
                task_label=task_label,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            return hidden_states, attention_mask, None
        
        batch_size, orig_seq_len = input_ids.shape
        device = input_ids.device
        
        # Get the initial input embeddings
        # We need to work with embeddings directly for appending virtual tokens
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        current_embeds = inputs_embeds  # [batch_size, seq_len, hidden_size]
        current_attention_mask = attention_mask.clone()
        
        # Track reasoning trajectory if requested
        trajectory = [] if return_trajectory else None
        
        # Keep track of image-related kwargs for the first pass only
        has_image = "pixel_values" in kwargs
        
        for step in range(reason_steps):
            # Forward pass to get hidden states
            if step == 0 and has_image:
                # First step with images: use input_ids to properly handle image tokens
                hidden_states = self.get_last_hidden_states(
                    task_label=task_label,
                    input_ids=input_ids,
                    attention_mask=current_attention_mask,
                    **kwargs,
                )
            else:
                # Subsequent steps or no images: use inputs_embeds
                # Remove image kwargs for subsequent steps
                non_image_kwargs = {k: v for k, v in kwargs.items() 
                                   if k not in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]}
                
                # Compute position_ids for the current sequence
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids=torch.zeros(batch_size, current_embeds.shape[1], dtype=torch.long, device=device),
                    image_grid_thw=None,
                    attention_mask=current_attention_mask,
                )
                
                non_image_kwargs["output_hidden_states"] = True
                outputs = super().forward(
                    task_label=task_label,
                    input_ids=None,
                    inputs_embeds=current_embeds,
                    attention_mask=current_attention_mask,
                    position_ids=position_ids,
                    rope_deltas=rope_deltas,
                    use_cache=False,
                    **non_image_kwargs,
                )
                hidden_states = outputs.hidden_states[-1]
            
            # Pool hidden states to get the "thought vector"
            pooled = torch.sum(
                hidden_states * current_attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(current_attention_mask, dim=1, keepdim=True)
            # Shape: [batch_size, hidden_size]
            
            if trajectory is not None:
                trajectory.append(pooled.clone())
            
            # Append the pooled vector as a virtual token
            virtual_token = pooled.unsqueeze(1)  # [batch_size, 1, hidden_size]
            current_embeds = torch.cat([current_embeds, virtual_token], dim=1)
            
            # Extend attention mask
            new_token_mask = torch.ones(batch_size, 1, device=device, dtype=current_attention_mask.dtype)
            current_attention_mask = torch.cat([current_attention_mask, new_token_mask], dim=1)
        
        # Final forward pass to get the output hidden states
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=torch.zeros(batch_size, current_embeds.shape[1], dtype=torch.long, device=device),
            image_grid_thw=None,
            attention_mask=current_attention_mask,
        )
        
        final_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]}
        final_kwargs["output_hidden_states"] = True
        
        outputs = super().forward(
            task_label=task_label,
            input_ids=None,
            inputs_embeds=current_embeds,
            attention_mask=current_attention_mask,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=False,
            **final_kwargs,
        )
        final_hidden_states = outputs.hidden_states[-1]
        
        return final_hidden_states, current_attention_mask, trajectory

    def get_last_hidden_states_with_cache(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any, torch.LongTensor]:
        """
        Get last hidden states with KV cache support for iterative reasoning.
        
        This method supports both:
        1. Initial pass: input_ids provided, returns hidden_states + new KV cache
        2. Incremental pass: inputs_embeds provided (virtual token), uses past_key_values
        
        Args:
            task_label: Task identifier for LoRA adapter selection
            input_ids: Input token IDs (for initial pass)
            attention_mask: Attention mask for the full sequence (including virtual tokens)
            past_key_values: Cached key-value pairs from previous passes
            inputs_embeds: Direct embeddings input (for virtual tokens in reasoning steps)
            cache_position: Position indices for the current tokens
            
        Returns:
            Tuple of (last_hidden_states, past_key_values, rope_deltas)
        """
        # Handle pixel_values padding removal
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)], dim=0
            )
        
        # Compute position_ids and rope_deltas
        if past_key_values is None:
            # Initial pass: compute full position_ids
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=kwargs.get("image_grid_thw", None),
                attention_mask=attention_mask,
            )
        else:
            # Incremental pass: compute position_ids for new token only
            # The position is simply the next position after the cached sequence
            batch_size = attention_mask.shape[0]
            seq_len = past_key_values[0][0].shape[2]  # Current cached sequence length
            
            # For incremental decoding, position_ids should be for the new token only
            # Shape: (3, batch_size, 1) for Qwen2.5-VL's 3D rope
            position_ids = torch.full(
                (3, batch_size, 1), 
                seq_len, 
                dtype=torch.long, 
                device=attention_mask.device
            )
            rope_deltas = kwargs.pop("rope_deltas", None)
        
        # Remove rope_deltas from kwargs if present (will be passed explicitly)
        kwargs.pop("rope_deltas", None)
        kwargs["output_hidden_states"] = True
        
        outputs = super().forward(
            task_label=task_label,
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=True,  # Enable KV cache
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise ValueError("Hidden states not found in model output")
        
        return hidden_states[-1], outputs.past_key_values, rope_deltas

    def get_reasoning_embeddings(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        reason_steps: int = 0,
        return_trajectory: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Perform multi-step implicit reasoning in the embedding space.
        
        This implements Chain-of-Thought in Vector Space by:
        1. Initial pass: Encode the input sequence with KV cache
        2. For each reasoning step:
           - Pool the current hidden states to get a "thought vector"
           - Use it as a virtual token embedding
           - Run incremental forward pass (using KV cache)
        3. Return final embeddings after all reasoning steps
        
        Args:
            task_label: Task identifier for LoRA adapter selection
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            reason_steps: Number of reasoning iterations (0 = no reasoning, original behavior)
            return_trajectory: Whether to return intermediate pooled embeddings
            **kwargs: Additional arguments (pixel_values, image_grid_thw, etc.)
            
        Returns:
            Tuple of (final_hidden_states, updated_attention_mask, reasoning_trajectory)
            - final_hidden_states: [batch_size, seq_len + reason_steps, hidden_size]
            - updated_attention_mask: [batch_size, seq_len + reason_steps]
            - reasoning_trajectory: List of pooled embeddings at each step (if return_trajectory=True)
        """
        ### 其实不会进入这一分支
        if reason_steps <= 0:
            # No reasoning, use original method
            hidden_states = self.get_last_hidden_states(
                task_label=task_label,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            return hidden_states, attention_mask, None
        
        batch_size, orig_seq_len = input_ids.shape
        device = input_ids.device

        # Step 0: Initial forward pass with KV cache
        hidden_states, past_key_values, rope_deltas = self.get_last_hidden_states_with_cache(
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        # Track reasoning trajectory if requested
        trajectory = [] if return_trajectory else None
        
        # Collect all hidden states for final output
        all_hidden_states = [hidden_states]  # [batch_size, seq_len, hidden_size]
        current_attention_mask = attention_mask.clone()
        
        # Remove image-related kwargs for reasoning steps (already processed)
        reasoning_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]}
        reasoning_kwargs["rope_deltas"] = rope_deltas
        
        for step in range(reason_steps):
            # Pool current hidden states to get the "thought vector"
            # Use mean pooling over valid tokens
            pooled = torch.sum(
                hidden_states * current_attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(current_attention_mask, dim=1, keepdim=True)
            # Shape: [batch_size, hidden_size]
            
            if trajectory is not None:
                trajectory.append(pooled.clone())
            
            # Use the pooled vector as a virtual token embedding
            # Shape: [batch_size, 1, hidden_size]
            virtual_token_embed = pooled.unsqueeze(1)
            
            # Extend attention mask for the new virtual token
            new_token_mask = torch.ones(batch_size, 1, device=device, dtype=current_attention_mask.dtype)
            extended_attention_mask = torch.cat([current_attention_mask, new_token_mask], dim=1)
            
            # Compute cache_position for the new token
            current_seq_len = extended_attention_mask.shape[1] - 1  # Position of new token
            cache_position = torch.tensor([current_seq_len], device=device, dtype=torch.long)
            
            # Incremental forward pass with the virtual token
            step_hidden_states, past_key_values, _ = self.get_last_hidden_states_with_cache(
                task_label=task_label,
                input_ids=None,  # Use inputs_embeds instead
                inputs_embeds=virtual_token_embed,
                attention_mask=extended_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **reasoning_kwargs,
            )
            # step_hidden_states shape: [batch_size, 1, hidden_size]
            ###？hiddenstates怎么append？拼接起来 也就是前面的hiddenstates加上这个新的hiddenstates
            # Append to collected hidden states
            all_hidden_states.append(step_hidden_states)
            current_attention_mask = extended_attention_mask
            
            # Update hidden_states for next iteration (full sequence for proper pooling)
            hidden_states = torch.cat(all_hidden_states, dim=1)
        
        # Concatenate all hidden states
        final_hidden_states = torch.cat(all_hidden_states, dim=1)
        # Shape: [batch_size, orig_seq_len + reason_steps, hidden_size]
        
        return final_hidden_states, current_attention_mask, trajectory

    def _init_projection_layer(self, config) -> None:
        """
        Initializes projection layers.
        """
        self.config.multi_vector_projector_dim = config.multi_vector_projector_dim

        self.multi_vector_projector = nn.Linear(
            in_features=self.config.text_config.hidden_size,
            out_features=self.config.multi_vector_projector_dim,
        )

    def get_single_vector_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Get the single-vector embeddings from the hidden states.
        """
        # if self._input_has_image(input_ids[0]):  # got document image
        #     img_start_positions = torch.where(
        #         input_ids == self.config.vision_start_token_id
        #     )[1]
        #     img_end_positions = torch.where(
        #         input_ids == self.config.vision_end_token_id
        #     )[1]

        #     batch_size, seq_len = input_ids.shape
        #     position_indices = torch.arange(seq_len, device=input_ids.device).expand(
        #         batch_size, -1
        #     )
        #     image_mask = (position_indices >= img_start_positions.unsqueeze(1)) & (
        #         position_indices <= img_end_positions.unsqueeze(1)
        #     )

        #     masked_hidden_states = hidden_states * image_mask.unsqueeze(-1)
        #     pooled_output = masked_hidden_states.sum(dim=1) / image_mask.sum(
        #         dim=1, keepdim=True
        #     )
        # else:  # got query text
        #     pooled_output = torch.sum(
        #         hidden_states * attention_mask.unsqueeze(-1), dim=1
        #     ) / torch.sum(attention_mask, dim=1, keepdim=True)

        ### 取消图像特殊处理，统一用attention_mask池化
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

        Only resizes and converts to tensor - NO normalization.
        This allows simple, lossless conversion back to PIL for Jina V4's native processing.

        Note: We use ToTensor() to satisfy UniIR collator's torch.stack() requirement.
        The conversion back to PIL via to_pil_image() is lossless since no normalization is applied.
        """
        from torchvision import transforms

        def img_preprocess_wrapper(image: Image.Image) -> torch.Tensor:
            target_size = getattr(self, 'mbeir_image_size', (224, 224))
            if isinstance(target_size, int):
                target_size = (target_size, target_size)

            # Only resize and convert to tensor - NO normalization
            # This allows lossless conversion back to PIL in encode_mbeir_batch()
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),  # Converts to [0, 1] range, required for collator's torch.stack()
            ])
            return transform(image)

        return img_preprocess_wrapper

    def get_tokenizer(self):
        """
        Returns a passthrough function that preserves raw text strings.

        The collator expects a callable that returns something with len().
        We return a simple wrapper class that stores raw strings for later
        processing by Jina V4's native process_texts() method.
        """


        def passthrough_wrapper(texts: List[str]) -> RawTextBatch:
            return RawTextBatch(texts)

        return passthrough_wrapper

    def encode_mbeir_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode a batch from UniIR MBEIR dataset using Jina V4's native processing.

        Handles three modality cases:
        - Text-only: txt_mask=1, image_mask=0
        - Image-only: txt_mask=0, image_mask=1
        - Multimodal: txt_mask=1, image_mask=1 (uses native image-text concatenation)

        Args:
            batch: Dictionary containing:
                - txt_batched: RawTextBatch containing raw strings (not tokenized)
                - image_batched: Image tensors [batch_size, 3, H, W] in [0, 1] range (not normalized)
                - txt_mask_batched: Text presence mask [batch_size]
                - image_mask_batched: Image presence mask [batch_size]
                - did_list or qid_list: List of hashed IDs

        Returns:
            Tuple of (embeddings tensor [batch_size, embed_dim], id_list)
        """
        ### encode_mbeir_batch 是eval进入的函数，所以必然有did_list or qid_list
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
        ### ？ 这里image_batched有放到cuda上吗？
        ### 应该是放了的，autocast也是在cuda上
        device = image_batched.device
        task_label = getattr(self, 'mbeir_task_label', 'retrieval')
        query_reason_steps = getattr(self, 'mbeir_reason_steps', 0)
        symmetric_qc_encoding = bool(getattr(self, 'mbeir_symmetric_reasoning', False))

        # Query/Candidate asymmetry control for evaluation embedding:
        # - asymmetric (default): query uses configured reason_steps, candidate uses 0
        # - symmetric: both query and candidate use configured reason_steps
        if is_query_batch and not is_candidate_batch:
            reason_steps = query_reason_steps
        elif is_candidate_batch and not is_query_batch:
            reason_steps = query_reason_steps if symmetric_qc_encoding else 0
        else:
            reason_steps = 0

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

        self._debug_dump_mbeir_batch_inputs(
            batch=batch,
            text_only_idx=text_only_idx,
            image_only_idx=image_only_idx,
            multimodal_idx=multimodal_idx,
        )

        embed_dim = self.config.text_config.hidden_size
        embeddings = torch.zeros(batch_size, embed_dim, device=device)

        # Process each modality group
        if text_only_idx:
            embeddings[text_only_idx] = self._encode_text_batch(txt_batched, text_only_idx, task_label, device, reason_steps)
        if image_only_idx:
            embeddings[image_only_idx] = self._encode_image_batch(image_batched, image_only_idx, task_label, device, reason_steps)
        if multimodal_idx:
            embeddings[multimodal_idx] = self._encode_multimodal_batch(txt_batched, image_batched, multimodal_idx, task_label, device, reason_steps)

        return embeddings, id_list

    def _encode_text_batch(
        self,
        txt_batched,  # RawTextBatch
        indices: List[int],
        task_label: str,
        device: torch.device,
        reason_steps: int = 0,
    ) -> torch.Tensor:
        """Encode text-only samples using Jina V4's native process_texts."""
        # Extract raw text strings from RawTextBatch
        texts = txt_batched[indices]

        # Use Jina V4's native text processing
        processed = self.processor.process_texts(texts)
        processed = {k: v.to(device) for k, v in processed.items()}

        output = self._forward_embeddings(task_label=task_label, reason_steps=reason_steps, **processed)
        return output.single_vec_emb

    def _encode_image_batch(
        self,
        image_batched: torch.Tensor,  # [B, 3, H, W], range [0, 1] (not normalized)
        indices: List[int],
        task_label: str,
        device: torch.device,
        reason_steps: int = 0,
    ) -> torch.Tensor:
        """Encode image-only samples using Jina V4's native process_images."""
        from torchvision.transforms.functional import to_pil_image

        # Convert unnormalized tensors to PIL (simple, no denormalization needed)
        pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in indices]

        # Use Jina V4's native image processing
        processed = self.processor.process_images(pil_images)
        processed = {k: v.to(device) for k, v in processed.items()}

        output = self._forward_embeddings(task_label=task_label, reason_steps=reason_steps, **processed)
        return output.single_vec_emb

    def _encode_multimodal_batch(
        self,
        txt_batched,  # RawTextBatch
        image_batched: torch.Tensor,  # [B, 3, H, W], range [0, 1] (not normalized)
        indices: List[int],
        task_label: str,
        device: torch.device,
        reason_steps: int = 0,
    ) -> torch.Tensor:
        """
        Encode multimodal samples using Jina V4's native process_multimodal.

        Uses format: <|vision_start|><|image_pad|><|vision_end|>{text}
        """
        from torchvision.transforms.functional import to_pil_image

        # Convert unnormalized tensors to PIL (simple, no denormalization needed)
        pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in indices]

        # Extract raw text strings from RawTextBatch (no decoding needed)
        texts = txt_batched[indices]

        # Use Jina V4's native multimodal processing
        processed = self.processor.process_multimodal(pil_images, texts)
        processed = {k: v.to(device) for k, v in processed.items()}

        output = self._forward_embeddings(task_label=task_label, reason_steps=reason_steps, **processed)
        return output.single_vec_emb
 
    ###原始方案 只拼接R个虚拟embedding
    def _forward_embeddings(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        reason_steps: int = 0,
        use_cache_for_reasoning: bool = False,
        **kwargs,
    ) -> JinaEmbeddingsV4ModelOutput:
        """
        Internal forward pass for embedding generation.
        This is the original forward logic, separated to avoid recursion with encode_mbeir_batch.
        
        Args:
            task_label: Task identifier for LoRA adapter selection
            input_ids: Input token IDs
            attention_mask: Attention mask
            reason_steps: Number of implicit reasoning steps (0 = disabled)
            use_cache_for_reasoning: If True, use KV cache for reasoning (faster, for inference).
                                     If False, use full forward passes (for training with gradients).
            **kwargs: Additional arguments (pixel_values, etc.)
        """
        self._debug_dump_encoding_inputs(
            source="_forward_embeddings",
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            reason_steps=reason_steps,
            kwargs=kwargs,
        )

        # Use reasoning-enhanced forward if reason_steps > 0
        if reason_steps > 0:
            if use_cache_for_reasoning:
                # Use KV cache version (fast, for inference)
                hidden_states, updated_attention_mask, trajectory = self.get_reasoning_embeddings(
                    task_label=task_label,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    reason_steps=reason_steps,
                    return_trajectory=False,
                    **kwargs,
                )
            else:
                # Use training-friendly version (no cache, proper gradients)
                hidden_states, updated_attention_mask, trajectory = self.get_reasoning_embeddings_for_training(
                    task_label=task_label,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    reason_steps=reason_steps,
                    return_trajectory=False,
                    **kwargs,
                )
            attention_mask = updated_attention_mask
        else:
            # Original behavior: single forward pass
            hidden_states = self.get_last_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_label=task_label,
                **kwargs,
            )

        # Compute the embeddings
        single_vec_emb = self.get_single_vector_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        multi_vec_emb = self.get_multi_vector_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            task_label=task_label,
        )

        return JinaEmbeddingsV4ModelOutput(
            vlm_last_hidden_states=None,
            single_vec_emb=single_vec_emb,
            multi_vec_emb=multi_vec_emb,
        )

    ### ==================== End MBEIR Interface Methods ====================

    def forward(
        self,
        task_label: Union[str, List[str]] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        output_vlm_last_hidden_states: bool = False,
        encode_mbeir_batch: bool = False,
        reason_steps: int = 0,
        return_reasoning_trajectory: bool = False,
        **kwargs,
    ) -> Union[JinaEmbeddingsV4ModelOutput, Tuple[torch.Tensor, List[int]]]:
        """
        Forward pass through the model. Returns both single-vector and multi-vector embeddings.
        
        Args:
            task_label: Task identifier for LoRA adapter selection
            input_ids (torch.Tensor): The input tokens tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            output_vlm_last_hidden_states: Whether to return hidden states
            encode_mbeir_batch: If True, use MBEIR batch encoding mode (kwargs contains batch dict)
            reason_steps: Number of implicit reasoning steps (0 = disabled, original behavior).
                         Each step pools the current hidden states and uses the result as a 
                         virtual token for incremental forward pass with KV cache.
            return_reasoning_trajectory: If True, return intermediate pooled embeddings from each step.
                         
        Returns:
            JinaEmbeddingsV4ModelOutput with:
                - single_vec_emb: Single-vector embeddings after reasoning
                - multi_vec_emb: Multi-vector embeddings after reasoning  
                - vlm_last_hidden_states: Hidden states (if output_vlm_last_hidden_states=True)
                - reasoning_trajectory: List of intermediate embeddings (if return_reasoning_trajectory=True)
            Or (embeddings, id_list) if encode_mbeir_batch=True
        """
        # Handle MBEIR batch encoding mode
        if encode_mbeir_batch:
            # kwargs contains the batch dict when called from mbeir_embedder
            return self.encode_mbeir_batch(kwargs)

        self._debug_dump_encoding_inputs(
            source="forward",
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            reason_steps=reason_steps,
            kwargs=kwargs,
        )

        # Use reasoning-enhanced forward if reason_steps > 0
        if reason_steps > 0:
            hidden_states, updated_attention_mask, trajectory = self.get_reasoning_embeddings(
                task_label=task_label,
                input_ids=input_ids,
                attention_mask=attention_mask,
                reason_steps=reason_steps,
                return_trajectory=return_reasoning_trajectory,
                **kwargs,
            )
            # Use updated attention mask for embedding computation
            attention_mask = updated_attention_mask
        else:
            # Original behavior: single forward pass
            hidden_states = self.get_last_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_label=task_label,
                **kwargs,
            )
            trajectory = None

        # Compute the embeddings
        single_vec_emb = self.get_single_vector_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
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
            reasoning_trajectory=trajectory,
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
        reason_steps: int = 0,
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
                    embeddings = self(**batch, task_label=task_label, reason_steps=reason_steps)
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
                        # When using reason_steps, attention_mask is extended
                        # We need to handle the case where embeddings have more tokens
                        if reason_steps > 0:
                            # Extend the mask to match the new sequence length
                            orig_len = valid_tokens.shape[1]
                            new_len = embeddings.shape[1]
                            if new_len > orig_len:
                                extra_mask = torch.ones(
                                    valid_tokens.shape[0], new_len - orig_len,
                                    dtype=valid_tokens.dtype, device=valid_tokens.device
                                )
                                valid_tokens = torch.cat([valid_tokens, extra_mask], dim=1)
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
        reason_steps: int = 0,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Encodes a list of texts into embeddings.

        Args:
            texts: text or list of text strings to encode
            max_length: Maximum token length for text processing
            batch_size: Number of texts to process at once
            return_multivector: Whether to return multi-vector embeddings instead of single-vector embeddings
            return_numpy: Whether to return numpy arrays instead of torch tensors
            truncate_dim: Dimension to truncate embeddings to (128, 256, 512, or 1024)
            prompt_name: Type of text being encoded ('query' or 'passage')
            reason_steps: Number of implicit reasoning steps in embedding space (0 = disabled).
                         Each step performs iterative refinement using KV cache.

        Returns:
            List of text embeddings as tensors or numpy arrays when encoding multiple texts, or single text embedding as tensor when encoding a single text
        """
        ### 默认为 “query” 前缀；做前缀/截断校验。
        prompt_name = prompt_name or "query"
        encode_kwargs = self._validate_encoding_params(
            truncate_dim=truncate_dim, prompt_name=prompt_name
        )

        task = self._validate_task(task)

        processor_fn = partial(
            self.processor.process_texts,
            max_length=max_length,
            prefix=encode_kwargs.pop("prefix"),
        )

        return_list = isinstance(texts, list)

        # If return_multivector is True and encoding multiple texts, ignore return_numpy
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
            reason_steps=reason_steps,
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
        reason_steps: int = 0,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Encodes a list of images or a single image into embedding(s).

        Args:
            images: image(s) to encode, can be PIL Image(s), URL(s), or local file path(s)
            batch_size: Number of images to process at once
            return_multivector: Whether to return multi-vector embeddings instead of single-vector embeddings
            return_numpy: Whether to return numpy arrays instead of torch tensors. If `return_multivector` is `True` and more than one image is encoded, this parameter is ignored.
            truncate_dim: Dimension to truncate embeddings to (128, 256, 512, or 1024)
            max_pixels: Maximum number of pixels to process per image
            reason_steps: Number of implicit reasoning steps in embedding space (0 = disabled).
                         Each step performs iterative refinement using KV cache.

        Returns:
            List of image embeddings as tensors or numpy arrays when encoding multiple images, or single image embedding as tensor when encoding a single image
        """
        ### max_pixels：临时覆盖处理器的图像像素上限（防止超大图OOM）
        ### 临时改写最大像素限制（编码结束会还原）。
        if max_pixels:
            default_max_pixels = self.processor.image_processor.max_pixels
            self.processor.image_processor.max_pixels = (
                max_pixels  # change during encoding
            )
        ### text中会有prefix，image没有
        encode_kwargs = self._validate_encoding_params(truncate_dim=truncate_dim)
        task = self._validate_task(task)

        return_list = isinstance(images, list)

        # If return_multivector is True and encoding multiple images, ignore return_numpy
        if return_multivector and return_list and len(images) > 1:
            if return_numpy:
                print(
                    "Warning: `return_numpy` is ignored when `return_multivector=True` and `len(images) > 1`"
                )
            return_numpy = False

        # Convert single image to list
        if isinstance(images, (str, Image.Image)):
            images = [images]

        images = self._load_images_if_needed(images)
        embeddings = self._process_batches(
            data=images,
            processor_fn=self.processor.process_images,
            desc="Encoding images...",
            task_label=task,
            batch_size=batch_size,
            return_multivector=return_multivector,
            return_numpy=return_numpy,
            reason_steps=reason_steps,
            **encode_kwargs,
        )

        ### 归还 max_pixels，保证线程安全/复用安全。
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
        Loads a pretrained model and configures it with the appropriate task adapter (`retrieval` by default).
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