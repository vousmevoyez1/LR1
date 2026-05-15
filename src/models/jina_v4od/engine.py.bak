# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.cuda.amp import autocast
# from torchvision.transforms.functional import to_pil_image
# from PIL import Image
# from typing import Dict, Any, List, Tuple, Optional

# from models.jina_v4 import utils


# def encode_batch_for_training(
#     model,
#     txt_batched,
#     image_batched: torch.Tensor,
#     txt_mask: torch.Tensor,
#     image_mask: torch.Tensor,
#     indices: List[int],
#     task_label: str,
#     device: torch.device,
#     reason_steps: int = 0,
#     use_cache_for_reasoning: bool = False,
#     max_token_length: Optional[int] = None,
# ) -> torch.Tensor:
#     """
#     Encode a subset of batch items using Jina V4's native processing.
    
#     Groups samples by modality (text-only, image-only, multimodal) and processes each group
#     with the appropriate method.
    
#     Args:
#         model: JinaEmbeddingsV4Model (unwrapped from DDP if needed)
#         txt_batched: RawTextBatch containing raw strings
#         image_batched: Image tensors [total_batch, 3, H, W] in [0, 1] range
#         txt_mask: Text presence mask [total_batch]
#         image_mask: Image presence mask [total_batch]
#         indices: Indices of items to encode from the flattened batch
#         task_label: Task identifier for LoRA adapter selection
#         device: Target device
#         reason_steps: Number of reasoning steps (0 = no reasoning)
#         use_cache_for_reasoning: If True, use KV cache (for inference). 
#                                   If False, use full forward (for training).
#         max_token_length: Maximum token length for truncation (None = use default)
        
#     Returns:
#         Tensor of embeddings [len(indices), embed_dim]
#     """
#     if len(indices) == 0:
#         embed_dim = model.config.text_config.hidden_size
#         return torch.zeros(0, embed_dim, device=device)
    
#     # Group indices by modality
#     text_only_idx, image_only_idx, multimodal_idx = [], [], []
#     for local_idx, global_idx in enumerate(indices):
#         has_text = txt_mask[global_idx].item() == 1
#         has_image = image_mask[global_idx].item() == 1
#         if has_text and has_image:
#             multimodal_idx.append((local_idx, global_idx))
#         elif has_image:
#             image_only_idx.append((local_idx, global_idx))
#         elif has_text:
#             text_only_idx.append((local_idx, global_idx))
    
#     embed_dim = model.config.text_config.hidden_size
#     embeddings = torch.zeros(len(indices), embed_dim, device=device)
    
#     # Process text-only samples
#     if text_only_idx:
#         local_indices = [x[0] for x in text_only_idx]
#         global_indices = [x[1] for x in text_only_idx]
#         texts = txt_batched[global_indices]
        
#         processed = model.processor.process_texts(texts, max_length=max_token_length)
#         processed = {k: v.to(device) for k, v in processed.items()}
        
#         output = model._forward_embeddings(
#             task_label=task_label, 
#             reason_steps=reason_steps, 
#             use_cache_for_reasoning=use_cache_for_reasoning,
#             **processed
#         )
#         embeddings[local_indices] = output.single_vec_emb
    
#     # Process image-only samples
#     if image_only_idx:
#         local_indices = [x[0] for x in image_only_idx]
#         global_indices = [x[1] for x in image_only_idx]
        
#         # Convert tensors to PIL images
#         pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in global_indices]
        
#         processed = model.processor.process_images(pil_images)
#         processed = {k: v.to(device) for k, v in processed.items()}
        
#         output = model._forward_embeddings(
#             task_label=task_label, 
#             reason_steps=reason_steps, 
#             use_cache_for_reasoning=use_cache_for_reasoning,
#             **processed
#         )
#         embeddings[local_indices] = output.single_vec_emb
    
#     # Process multimodal samples
#     if multimodal_idx:
#         local_indices = [x[0] for x in multimodal_idx]
#         global_indices = [x[1] for x in multimodal_idx]
        
#         pil_images = [to_pil_image(image_batched[idx].cpu()) for idx in global_indices]
#         texts = txt_batched[global_indices]
        
#         processed = model.processor.process_multimodal(pil_images, texts, max_length=max_token_length)
#         processed = {k: v.to(device) for k, v in processed.items()}
        
#         output = model._forward_embeddings(
#             task_label=task_label, 
#             reason_steps=reason_steps, 
#             use_cache_for_reasoning=use_cache_for_reasoning,
#             **processed
#         )
#         embeddings[local_indices] = output.single_vec_emb
    
#     return embeddings


# def compute_contrastive_loss(
#     query_embeds: torch.Tensor,
#     pos_cand_embeds: torch.Tensor,
#     neg_cand_embeds: Optional[torch.Tensor] = None,
#     temperature: float = 0.07,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Compute InfoNCE contrastive loss.
    
#     Uses in-batch negatives plus optional hard negatives.
    
#     Args:
#         query_embeds: Query embeddings [batch_size, embed_dim]
#         pos_cand_embeds: Positive candidate embeddings [batch_size, embed_dim]
#         neg_cand_embeds: Optional hard negative embeddings [batch_size * num_neg, embed_dim]
#         temperature: Temperature for softmax scaling
        
#     Returns:
#         Tuple of (loss, accuracy)
#     """
#     batch_size = query_embeds.size(0)
#     device = query_embeds.device
    
#     # Normalize embeddings (already normalized from model, but ensure)
#     query_embeds = F.normalize(query_embeds, p=2, dim=-1)
#     pos_cand_embeds = F.normalize(pos_cand_embeds, p=2, dim=-1)
    
#     # Build candidate pool: [pos_cands, in_batch_negs, hard_negs]
#     # In-batch negatives: other positives in the batch serve as negatives
#     # For each query, its positive is at the diagonal
    
#     # Similarity matrix: [batch_size, batch_size]
#     # sim[i, j] = query[i] · pos_cand[j]
#     sim_matrix = torch.matmul(query_embeds, pos_cand_embeds.t()) / temperature
    
#     # Add hard negatives if available
#     if neg_cand_embeds is not None and neg_cand_embeds.size(0) > 0:
#         neg_cand_embeds = F.normalize(neg_cand_embeds, p=2, dim=-1)
#         num_neg_per_query = neg_cand_embeds.size(0) // batch_size
#         neg_cand_embeds = neg_cand_embeds.view(batch_size, num_neg_per_query, -1)
        
#         # [batch_size, num_neg]
#         neg_sim = torch.bmm(query_embeds.unsqueeze(1), neg_cand_embeds.transpose(1, 2)).squeeze(1) / temperature
        
#         # Concatenate: [batch_size, batch_size + num_neg]
#         logits = torch.cat([sim_matrix, neg_sim], dim=1)
#     else:
#         logits = sim_matrix
    
#     # Labels: positive is at position i for query i (diagonal)
#     labels = torch.arange(batch_size, device=device)
    
#     # Cross-entropy loss
#     loss = F.cross_entropy(logits, labels)
    
#     # Compute accuracy
#     predictions = logits.argmax(dim=1)
#     accuracy = (predictions == labels).float().mean()
    
#     return loss, accuracy


# def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
#     """
#     Train Jina V4 model for one epoch using contrastive loss.
    
#     The model is a PeftModel wrapping JinaEmbeddingsV4Model. Only LoRA parameters are trained.
#     Uses single-vector embeddings for contrastive learning.
#     """
#     model.train()
    
#     # Get the base model (unwrap DDP and PEFT if needed)
#     if hasattr(model, 'module'):
#         model_without_ddp = model.module
#     else:
#         model_without_ddp = model
    
#     # For PEFT model, get the base model for encoding
#     if hasattr(model_without_ddp, 'base_model'):
#         base_model = model_without_ddp.base_model.model
#     else:
#         base_model = model_without_ddp

#     # Metric logger setup
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     header = "Train Epoch: [{}]".format(epoch)
#     print_freq = config.trainer_config.print_freq

#     accumulation_steps = config.trainer_config.gradient_accumulation_steps
#     accumulation_counter = 0
    
#     # Task label for LoRA adapter selection (default to 'retrieval')
#     task_label = getattr(config.model, 'task_label', 'retrieval')
#     temperature = getattr(config.model, 'temperature', 0.07)
#     # Reasoning steps: 0 = no reasoning, >0 = multi-step reasoning
#     reason_steps = getattr(config.model, 'reason_steps', 0)
#     # Max token length for truncation (helps increase batch size)
#     max_token_length = getattr(config.model, 'max_token_length', None)
    
#     if reason_steps > 0:
#         print(f"Training with reason_steps={reason_steps} (using training-friendly reasoning without KV cache)")
#     if max_token_length is not None:
#         print(f"Using max_token_length={max_token_length} for input truncation")
    
#     for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         # Move tensors to GPU
#         txt_batched = batch["txt_batched"]  # RawTextBatch
#         image_batched = batch["image_batched"].to(gpu_id, non_blocking=True)
#         txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
#         image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)
#         index_mapping = batch["index_mapping"]
        
#         batch_size = len(index_mapping["query"])
        
#         # Flatten indices for query, pos_cand, neg_cand
#         query_indices = [idx[0] for idx in index_mapping["query"]]  # Each query has one index
#         pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]
        
#         # Handle negative candidates if present
#         neg_cand_indices = []
#         if "neg_cand_list" in index_mapping:
#             for neg_list in index_mapping["neg_cand_list"]:
#                 neg_cand_indices.extend(neg_list)

#         # autocast for mixed precision
#         with autocast():
#             # Encode queries (with reasoning if configured)
#             # use_cache_for_reasoning=False for training (proper gradient flow)
#             query_embeds = encode_batch_for_training(
#                 model=base_model,
#                 txt_batched=txt_batched,
#                 image_batched=image_batched,
#                 txt_mask=txt_mask,
#                 image_mask=image_mask,
#                 indices=query_indices,
#                 task_label=task_label,
#                 device=gpu_id,
#                 reason_steps=reason_steps,
#                 use_cache_for_reasoning=False,  # Training: no KV cache
#                 max_token_length=max_token_length,
#             )
            
#             # Encode positive candidates (candidates typically don't need reasoning)
#             pos_cand_embeds = encode_batch_for_training(
#                 model=base_model,
#                 txt_batched=txt_batched,
#                 image_batched=image_batched,
#                 txt_mask=txt_mask,
#                 image_mask=image_mask,
#                 indices=pos_cand_indices,
#                 task_label=task_label,
#                 device=gpu_id,
#                 reason_steps=0,  # Candidates don't use reasoning
#                 use_cache_for_reasoning=False,
#                 max_token_length=max_token_length,
#             )
            
#             # Encode negative candidates if present
#             neg_cand_embeds = None
#             if neg_cand_indices:
#                 neg_cand_embeds = encode_batch_for_training(
#                     model=base_model,
#                     txt_batched=txt_batched,
#                     image_batched=image_batched,
#                     txt_mask=txt_mask,
#                     image_mask=image_mask,
#                     indices=neg_cand_indices,
#                     task_label=task_label,
#                     device=gpu_id,
#                     reason_steps=0,  # Candidates don't use reasoning
#                     use_cache_for_reasoning=False,
#                     max_token_length=max_token_length,
#                 )
            
#             # Compute contrastive loss
#             loss, inbatch_accuracy = compute_contrastive_loss(
#                 query_embeds=query_embeds,
#                 pos_cand_embeds=pos_cand_embeds,
#                 neg_cand_embeds=neg_cand_embeds,
#                 temperature=temperature,
#             )

#         # Scale the loss for gradient accumulation
#         loss = loss / accumulation_steps

#         # AMP backward pass
#         scaler.scale(loss).backward()

#         accumulation_counter += 1
#         if accumulation_counter == accumulation_steps:
#             global_step += 1

#             # Optimizer step with scaler
#             scaler.step(optimizer)
#             scaler.update()

#             optimizer.zero_grad(set_to_none=True)
#             scheduler.step()
#             accumulation_counter = 0

#         metric_logger.update(loss=loss.item() * accumulation_steps)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

#     # Gather stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger.global_avg())
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
#     """
#     Evaluate Jina V4 model using contrastive loss metrics.
    
#     Unlike the old version, this doesn't use queues since JinaEmbeddingsV4Model
#     doesn't have queue-based momentum contrast.
#     """
#     model.eval()
    
#     # Get the base model for encoding
#     if hasattr(model_without_ddp, 'base_model'):
#         base_model = model_without_ddp.base_model.model
#     else:
#         base_model = model_without_ddp

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     header = "Test:"
#     print_freq = config.evaluator.print_freq

#     # Task label for LoRA adapter selection
#     task_label = getattr(config.model, 'task_label', 'retrieval')
#     temperature = getattr(config.model, 'temperature', 0.07)
#     reason_steps = getattr(config.model, 'reason_steps', 0)
#     # In evaluation, we can use KV cache for efficiency (no gradients needed)
#     use_cache_for_reasoning = getattr(config.model, 'use_cache_for_eval', True)
    
#     if reason_steps > 0:
#         cache_mode = "with KV cache" if use_cache_for_reasoning else "without KV cache"
#         print(f"Evaluating with reason_steps={reason_steps} ({cache_mode})")

#     for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         # Move tensors to GPU
#         txt_batched = batch["txt_batched"]  # RawTextBatch
#         image_batched = batch["image_batched"].to(gpu_id, non_blocking=True)
#         txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
#         image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)
#         index_mapping = batch["index_mapping"]
        
#         batch_size = len(index_mapping["query"])
        
#         # Flatten indices
#         query_indices = [idx[0] for idx in index_mapping["query"]]
#         pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]

#         # autocast for mixed precision
#         with autocast():
#             # Encode queries (with reasoning if configured)
#             query_embeds = encode_batch_for_training(
#                 model=base_model,
#                 txt_batched=txt_batched,
#                 image_batched=image_batched,
#                 txt_mask=txt_mask,
#                 image_mask=image_mask,
#                 indices=query_indices,
#                 task_label=task_label,
#                 device=gpu_id,
#                 reason_steps=reason_steps,
#                 use_cache_for_reasoning=use_cache_for_reasoning,
#             )
            
#             # Encode positive candidates (no reasoning for candidates)
#             pos_cand_embeds = encode_batch_for_training(
#                 model=base_model,
#                 txt_batched=txt_batched,
#                 image_batched=image_batched,
#                 txt_mask=txt_mask,
#                 image_mask=image_mask,
#                 indices=pos_cand_indices,
#                 task_label=task_label,
#                 device=gpu_id,
#                 reason_steps=0,
#                 use_cache_for_reasoning=use_cache_for_reasoning,
#             )
            
#             # Compute contrastive loss (in-batch only for evaluation)
#             loss, inbatch_accuracy = compute_contrastive_loss(
#                 query_embeds=query_embeds,
#                 pos_cand_embeds=pos_cand_embeds,
#                 neg_cand_embeds=None,
#                 temperature=temperature,
#             )

#         metric_logger.update(loss=loss.item())
#         metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

#     # Gather stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger.global_avg())

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

from models.jina_v4 import utils


def encode_batch_for_training(
    model,
    txt_batched,
    image_batched: torch.Tensor,
    txt_mask: torch.Tensor,
    image_mask: torch.Tensor,
    indices: List[int],
    task_label: str,
    device: torch.device,
    reason_steps: int = 0,
    use_cache_for_reasoning: bool = False,
    max_token_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Encode a subset of batch items using Jina V4's native processing.
    
    Groups samples by modality (text-only, image-only, multimodal) and processes each group
    with the appropriate method.
    
    Args:
        model: JinaEmbeddingsV4Model (unwrapped from DDP if needed)
        txt_batched: RawTextBatch containing raw strings
        image_batched: Image tensors [total_batch, 3, H, W] in [0, 1] range (SHOULD BE ON CPU)
        txt_mask: Text presence mask [total_batch] (SHOULD BE ON GPU)
        image_mask: Image presence mask [total_batch] (SHOULD BE ON GPU)
        indices: Indices of items to encode from the flattened batch
        task_label: Task identifier for LoRA adapter selection
        device: Target device
        reason_steps: Number of reasoning steps (0 = no reasoning)
        use_cache_for_reasoning: If True, use KV cache (for inference). 
                                  If False, use full forward (for training).
        max_token_length: Maximum token length for truncation (None = use default)
        
    Returns:
        Tensor of embeddings [len(indices), embed_dim]
    """
    if len(indices) == 0:
        embed_dim = model.config.text_config.hidden_size
        return torch.zeros(0, embed_dim, device=device)
    
    # Group indices by modality
    text_only_idx, image_only_idx, multimodal_idx = [], [], []
    for local_idx, global_idx in enumerate(indices):
        has_text = txt_mask[global_idx].item() == 1
        has_image = image_mask[global_idx].item() == 1
        if has_text and has_image:
            multimodal_idx.append((local_idx, global_idx))
        elif has_image:
            image_only_idx.append((local_idx, global_idx))
        elif has_text:
            text_only_idx.append((local_idx, global_idx))
    
    embed_dim = model.config.text_config.hidden_size
    embeddings = torch.zeros(len(indices), embed_dim, device=device)
    
    # Process text-only samples
    if text_only_idx:
        local_indices = [x[0] for x in text_only_idx]
        global_indices = [x[1] for x in text_only_idx]
        texts = txt_batched[global_indices]
        
        processed = model.processor.process_texts(texts, max_length=max_token_length)
        processed = {k: v.to(device) for k, v in processed.items()}
        
        output = model._forward_embeddings(
            task_label=task_label, 
            reason_steps=reason_steps, 
            use_cache_for_reasoning=use_cache_for_reasoning,
            **processed
        )
        embeddings[local_indices] = output.single_vec_emb
    
    # Process image-only samples
    if image_only_idx:
        local_indices = [x[0] for x in image_only_idx]
        global_indices = [x[1] for x in image_only_idx]
        
        # Convert tensors to PIL images
        # OPTIMIZATION: image_batched is on CPU, so we remove .cpu() call to avoid overhead
        pil_images = [to_pil_image(image_batched[idx]) for idx in global_indices]
        
        processed = model.processor.process_images(pil_images)
        processed = {k: v.to(device) for k, v in processed.items()}
        
        output = model._forward_embeddings(
            task_label=task_label, 
            reason_steps=reason_steps, 
            use_cache_for_reasoning=use_cache_for_reasoning,
            **processed
        )
        embeddings[local_indices] = output.single_vec_emb
    
    # Process multimodal samples
    if multimodal_idx:
        local_indices = [x[0] for x in multimodal_idx]
        global_indices = [x[1] for x in multimodal_idx]
        
        # OPTIMIZATION: image_batched is on CPU, remove .cpu()
        pil_images = [to_pil_image(image_batched[idx]) for idx in global_indices]
        texts = txt_batched[global_indices]
        
        processed = model.processor.process_multimodal(pil_images, texts, max_length=max_token_length)
        processed = {k: v.to(device) for k, v in processed.items()}
        
        output = model._forward_embeddings(
            task_label=task_label, 
            reason_steps=reason_steps, 
            use_cache_for_reasoning=use_cache_for_reasoning,
            **processed
        )
        embeddings[local_indices] = output.single_vec_emb
    
    return embeddings


def compute_contrastive_loss(
    query_embeds: torch.Tensor,
    pos_cand_embeds: torch.Tensor,
    neg_cand_embeds: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute InfoNCE contrastive loss.
    
    Uses in-batch negatives plus optional hard negatives.
    
    Args:
        query_embeds: Query embeddings [batch_size, embed_dim]
        pos_cand_embeds: Positive candidate embeddings [batch_size, embed_dim]
        neg_cand_embeds: Optional hard negative embeddings [batch_size * num_neg, embed_dim]
        temperature: Temperature for softmax scaling
        
    Returns:
        Tuple of (loss, accuracy)
    """
    batch_size = query_embeds.size(0)
    device = query_embeds.device
    
    # Normalize embeddings (already normalized from model, but ensure)
    query_embeds = F.normalize(query_embeds, p=2, dim=-1)
    pos_cand_embeds = F.normalize(pos_cand_embeds, p=2, dim=-1)
    
    # Build candidate pool: [pos_cands, in_batch_negs, hard_negs]
    # In-batch negatives: other positives in the batch serve as negatives
    # For each query, its positive is at the diagonal
    
    # Similarity matrix: [batch_size, batch_size]
    # sim[i, j] = query[i] · pos_cand[j]
    sim_matrix = torch.matmul(query_embeds, pos_cand_embeds.t()) / temperature
    
    # Add hard negatives if available
    if neg_cand_embeds is not None and neg_cand_embeds.size(0) > 0:
        neg_cand_embeds = F.normalize(neg_cand_embeds, p=2, dim=-1)
        num_neg_per_query = neg_cand_embeds.size(0) // batch_size
        neg_cand_embeds = neg_cand_embeds.view(batch_size, num_neg_per_query, -1)
        
        # [batch_size, num_neg]
        neg_sim = torch.bmm(query_embeds.unsqueeze(1), neg_cand_embeds.transpose(1, 2)).squeeze(1) / temperature
        
        # Concatenate: [batch_size, batch_size + num_neg]
        logits = torch.cat([sim_matrix, neg_sim], dim=1)
    else:
        logits = sim_matrix
    
    # Labels: positive is at position i for query i (diagonal)
    labels = torch.arange(batch_size, device=device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    # Compute accuracy
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).float().mean()
    
    return loss, accuracy


def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    """
    Train Jina V4 model for one epoch using contrastive loss.
    
    The model is a PeftModel wrapping JinaEmbeddingsV4Model. Only LoRA parameters are trained.
    Uses single-vector embeddings for contrastive learning.
    
    Returns:
        Tuple of (train_stats dict, updated global_step)
    """
    model.train()
    
    # Get the base model (unwrap DDP and PEFT if needed)
    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # For PEFT model, get the base model for encoding
    if hasattr(model_without_ddp, 'base_model'):
        base_model = model_without_ddp.base_model.model
    else:
        base_model = model_without_ddp

    # Metric logger setup
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0
    
    # Task label for LoRA adapter selection (default to 'retrieval')
    task_label = getattr(config.model, 'task_label', 'retrieval')
    temperature = getattr(config.model, 'temperature', 0.07)
    # Reasoning steps: 0 = no reasoning, >0 = multi-step reasoning
    reason_steps = getattr(config.model, 'reason_steps', 0)
    # Max token length for truncation (helps increase batch size)
    max_token_length = getattr(config.model, 'max_token_length', None)
    # Step-level checkpoint saving (0 = disabled)
    save_steps = getattr(config.trainer_config, 'save_steps', 0)
    
    if reason_steps > 0:
        print(f"Training with reason_steps={reason_steps} (using training-friendly reasoning without KV cache)")
    if max_token_length is not None:
        print(f"Using max_token_length={max_token_length} for input truncation")
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move tensors to GPU
        txt_batched = batch["txt_batched"]  # RawTextBatch
        
        # OPTIMIZATION: Do NOT move image_batched to GPU here.
        # It is handled in CPU RAM and converted to PIL in encode_batch_for_training
        # This prevents GPU-CPU-GPU ping-pong and saves huge VRAM/Bandwidth
        image_batched = batch["image_batched"] 
        
        # Move masks to GPU as they are needed for indexing
        txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
        image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)
        index_mapping = batch["index_mapping"]
        
        batch_size = len(index_mapping["query"])
        
        # Flatten indices for query, pos_cand, neg_cand
        query_indices = [idx[0] for idx in index_mapping["query"]]  # Each query has one index
        pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]
        
        # Handle negative candidates if present
        neg_cand_indices = []
        if "neg_cand_list" in index_mapping:
            for neg_list in index_mapping["neg_cand_list"]:
                neg_cand_indices.extend(neg_list)

        # autocast for mixed precision
        with autocast(dtype=torch.bfloat16):
            # Encode queries (with reasoning if configured)
            # use_cache_for_reasoning=False for training (proper gradient flow)
            query_embeds = encode_batch_for_training(
                model=base_model,
                txt_batched=txt_batched,
                image_batched=image_batched, # Passes CPU tensor
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=query_indices,
                task_label=task_label,
                device=gpu_id,
                reason_steps=reason_steps,
                use_cache_for_reasoning=False,  # Training: no KV cache
                max_token_length=max_token_length,
            )
            
            # Encode positive candidates (candidates typically don't need reasoning)
            pos_cand_embeds = encode_batch_for_training(
                model=base_model,
                txt_batched=txt_batched,
                image_batched=image_batched, # Passes CPU tensor
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=pos_cand_indices,
                task_label=task_label,
                device=gpu_id,
                reason_steps=0,  # Candidates don't use reasoning
                use_cache_for_reasoning=False,
                max_token_length=max_token_length,
            )
            
            # Encode negative candidates if present
            neg_cand_embeds = None
            if neg_cand_indices:
                neg_cand_embeds = encode_batch_for_training(
                    model=base_model,
                    txt_batched=txt_batched,
                    image_batched=image_batched, # Passes CPU tensor
                    txt_mask=txt_mask,
                    image_mask=image_mask,
                    indices=neg_cand_indices,
                    task_label=task_label,
                    device=gpu_id,
                    reason_steps=0,  # Candidates don't use reasoning
                    use_cache_for_reasoning=False,
                    max_token_length=max_token_length,
                )
            
            # Compute contrastive loss
            loss, inbatch_accuracy = compute_contrastive_loss(
                query_embeds=query_embeds,
                pos_cand_embeds=pos_cand_embeds,
                neg_cand_embeds=neg_cand_embeds,
                temperature=temperature,
            )

        # Scale the loss for gradient accumulation
        loss = loss / accumulation_steps

        # AMP backward pass
        #scaler.scale(loss).backward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # Optimizer step with scaler
            # scaler.step(optimizer)
            # scaler.update()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulation_counter = 0
            
            # Step-level checkpoint saving
            if save_steps > 0 and global_step % save_steps == 0:
                if utils.is_main_process():
                    from models.jina_v4.train import save_step_checkpoint
                    save_step_checkpoint(
                        model_without_ddp, optimizer, scheduler, 
                        epoch, global_step, scaler, config
                    )

        metric_logger.update(loss=loss.item() * accumulation_steps)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # 确保所有 rank 在同步统计数据前都完成了训练循环
    # 这可以防止因数据不均衡导致的 NCCL 超时
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    
    # Return both stats and global_step for tracking
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step


@torch.no_grad()
def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
    """
    Evaluate Jina V4 model using contrastive loss metrics.
    
    Unlike the old version, this doesn't use queues since JinaEmbeddingsV4Model
    doesn't have queue-based momentum contrast.
    """
    model.eval()
    
    # Get the base model for encoding
    if hasattr(model_without_ddp, 'base_model'):
        base_model = model_without_ddp.base_model.model
    else:
        base_model = model_without_ddp

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    # Task label for LoRA adapter selection
    task_label = getattr(config.model, 'task_label', 'retrieval')
    temperature = getattr(config.model, 'temperature', 0.07)
    reason_steps = getattr(config.model, 'reason_steps', 0)
    # In evaluation, we can use KV cache for efficiency (no gradients needed)
    use_cache_for_reasoning = getattr(config.model, 'use_cache_for_eval', True)
    
    if reason_steps > 0:
        cache_mode = "with KV cache" if use_cache_for_reasoning else "without KV cache"
        print(f"Evaluating with reason_steps={reason_steps} ({cache_mode})")

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move tensors to GPU
        txt_batched = batch["txt_batched"]  # RawTextBatch
        
        # OPTIMIZATION: Do NOT move image_batched to GPU here (Same as training)
        image_batched = batch["image_batched"]
        
        txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
        image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)
        index_mapping = batch["index_mapping"]
        
        batch_size = len(index_mapping["query"])
        
        # Flatten indices
        query_indices = [idx[0] for idx in index_mapping["query"]]
        pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]

        # autocast for mixed precision
        with autocast():
            # Encode queries (with reasoning if configured)
            query_embeds = encode_batch_for_training(
                model=base_model,
                txt_batched=txt_batched,
                image_batched=image_batched, # Passes CPU tensor
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=query_indices,
                task_label=task_label,
                device=gpu_id,
                reason_steps=reason_steps,
                use_cache_for_reasoning=use_cache_for_reasoning,
            )
            
            # Encode positive candidates (no reasoning for candidates)
            pos_cand_embeds = encode_batch_for_training(
                model=base_model,
                txt_batched=txt_batched,
                image_batched=image_batched, # Passes CPU tensor
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=pos_cand_indices,
                task_label=task_label,
                device=gpu_id,
                reason_steps=0,
                use_cache_for_reasoning=use_cache_for_reasoning,
            )
            
            # Compute contrastive loss (in-batch only for evaluation)
            loss, inbatch_accuracy = compute_contrastive_loss(
                query_embeds=query_embeds,
                pos_cand_embeds=pos_cand_embeds,
                neg_cand_embeds=None,
                temperature=temperature,
            )

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # 确保所有 rank 在同步统计数据前都完成了评估循环
    # 这可以防止因数据不均衡导致的 NCCL 超时
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}