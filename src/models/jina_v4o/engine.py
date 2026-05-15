"""
Engine module for Jina V4 with Thought Tokens and Deep Supervision.

This module implements the training loop with:
1. Thought Tokens (Reasoning Anchors): <thought_1>...<thought_K> appended to inputs
2. Deep Supervision: Loss computed at each thought token position
3. Adaptive gamma scheduling: γ increases from gamma_start to gamma_end during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

import models.jina_v4o.utils as utils


def encode_batch_for_training(
    model,
    txt_batched,
    image_batched: torch.Tensor,
    txt_mask: torch.Tensor,
    image_mask: torch.Tensor,
    indices: List[int],
    task_label: str,
    device: torch.device,
    num_thought_tokens: int = 0,
    thought_pooling_mode: str = "last",
    return_thought_embeddings: bool = False,
    max_token_length: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """
    Encode a subset of batch items using Jina V4's native processing with thought tokens.
    
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
        num_thought_tokens: Number of thought tokens (K) to use (0 = disabled)
        thought_pooling_mode: How to build final embedding when thought tokens are enabled.
            - "last": use <thought_K> embedding (default)
            - "mean_all_thought_tokens": mean-pool all thought token embeddings
        return_thought_embeddings: If True, return embeddings at each thought position
        max_token_length: Maximum token length for truncation (None = use default)
        
    Returns:
        Tuple of:
            - embeddings: Tensor of embeddings [len(indices), embed_dim]
            - thought_embeddings: List of K tensors [len(indices), embed_dim] or None
    """
    if len(indices) == 0:
        embed_dim = model.config.text_config.hidden_size
        empty_emb = torch.zeros(0, embed_dim, device=device)
        return empty_emb, None if return_thought_embeddings else empty_emb
    
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
    
    # Initialize thought embeddings storage if needed
    thought_embeddings_list = None
    if return_thought_embeddings and num_thought_tokens > 0:
        thought_embeddings_list = [
            torch.zeros(len(indices), embed_dim, device=device) 
            for _ in range(num_thought_tokens)
        ]
    
    # Process text-only samples
    if text_only_idx:
        local_indices = [x[0] for x in text_only_idx]
        global_indices = [x[1] for x in text_only_idx]
        texts = txt_batched[global_indices]
        
        processed = model.processor.process_texts(
            texts, 
            max_length=max_token_length,
            num_thought_tokens=num_thought_tokens
        )
        processed = {k: v.to(device) for k, v in processed.items()}
        
        output = model._forward_embeddings(
            task_label=task_label, 
            num_thought_tokens=num_thought_tokens,
            thought_pooling_mode=thought_pooling_mode,
            return_thought_embeddings=return_thought_embeddings,
            **processed
        )
        embeddings[local_indices] = output.single_vec_emb
        
        if return_thought_embeddings and output.thought_embeddings is not None:
            for k, thought_emb in enumerate(output.thought_embeddings):
                thought_embeddings_list[k][local_indices] = thought_emb
    
    # Process image-only samples
    if image_only_idx:
        local_indices = [x[0] for x in image_only_idx]
        global_indices = [x[1] for x in image_only_idx]
        
        # Convert tensors to PIL images (image_batched is on CPU)
        pil_images = [to_pil_image(image_batched[idx]) for idx in global_indices]
        
        processed = model.processor.process_images(
            pil_images,
            num_thought_tokens=num_thought_tokens
        )
        processed = {k: v.to(device) for k, v in processed.items()}
        
        output = model._forward_embeddings(
            task_label=task_label, 
            num_thought_tokens=num_thought_tokens,
            thought_pooling_mode=thought_pooling_mode,
            return_thought_embeddings=return_thought_embeddings,
            **processed
        )
        embeddings[local_indices] = output.single_vec_emb
        
        if return_thought_embeddings and output.thought_embeddings is not None:
            for k, thought_emb in enumerate(output.thought_embeddings):
                thought_embeddings_list[k][local_indices] = thought_emb
    
    # Process multimodal samples
    if multimodal_idx:
        local_indices = [x[0] for x in multimodal_idx]
        global_indices = [x[1] for x in multimodal_idx]
        
        pil_images = [to_pil_image(image_batched[idx]) for idx in global_indices]
        texts = txt_batched[global_indices]
        
        processed = model.processor.process_multimodal(
            pil_images, 
            texts, 
            max_length=max_token_length,
            num_thought_tokens=num_thought_tokens
        )
        processed = {k: v.to(device) for k, v in processed.items()}
        
        output = model._forward_embeddings(
            task_label=task_label, 
            num_thought_tokens=num_thought_tokens,
            thought_pooling_mode=thought_pooling_mode,
            return_thought_embeddings=return_thought_embeddings,
            **processed
        )
        embeddings[local_indices] = output.single_vec_emb
        
        if return_thought_embeddings and output.thought_embeddings is not None:
            for k, thought_emb in enumerate(output.thought_embeddings):
                thought_embeddings_list[k][local_indices] = thought_emb
    
    if return_thought_embeddings:
        return embeddings, thought_embeddings_list
    return embeddings, None


def compute_infonce_loss(
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
    
    # Normalize embeddings
    query_embeds = F.normalize(query_embeds, p=2, dim=-1)
    pos_cand_embeds = F.normalize(pos_cand_embeds, p=2, dim=-1)
    
    # Similarity matrix: [batch_size, batch_size]
    sim_matrix = torch.matmul(query_embeds, pos_cand_embeds.t()) / temperature
    
    # Add hard negatives if available
    if neg_cand_embeds is not None and neg_cand_embeds.size(0) > 0:
        neg_cand_embeds = F.normalize(neg_cand_embeds, p=2, dim=-1)
        num_neg_per_query = neg_cand_embeds.size(0) // batch_size
        neg_cand_embeds = neg_cand_embeds.view(batch_size, num_neg_per_query, -1)
        
        # [batch_size, num_neg]
        neg_sim = torch.bmm(
            query_embeds.unsqueeze(1), 
            neg_cand_embeds.transpose(1, 2)
        ).squeeze(1) / temperature
        
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


def compute_deep_supervision_loss(
    query_thought_embeddings: List[torch.Tensor],  # K embeddings, each [batch, dim]
    pos_cand_embeds: torch.Tensor,  # [batch, dim]
    neg_cand_embeds: Optional[torch.Tensor],
    temperature: float = 0.07,
    gamma: float = 0.5,
    deep_supervision_lambda: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute deep supervision loss with adaptive weighting.
    
    Loss function:
        L = InfoNCE(e_K, Doc+) + λ * Σ_{i=1}^{K-1} γ^{K-i} * InfoNCE(e_i, Doc+)
    
    This ensures:
    1. The final thought token (e_K) receives the strongest supervision
    2. Earlier thought tokens receive progressively smaller supervision
    3. γ controls how much we care about intermediate steps
    
    Args:
        query_thought_embeddings: List of K embeddings from thought tokens
        pos_cand_embeds: Positive candidate embeddings [batch, dim]
        neg_cand_embeds: Optional hard negatives
        temperature: Softmax temperature
        gamma: Decay factor for intermediate losses (0.1 to 0.8)
        deep_supervision_lambda: Weight for auxiliary losses
        
    Returns:
        Tuple of (total_loss, accuracy, loss_dict with individual losses)
    """
    K = len(query_thought_embeddings)
    
    if K == 0:
        # Fallback: no thought tokens, shouldn't happen
        raise ValueError("No thought embeddings provided for deep supervision")
    
    # Main loss: from the last thought token (e_K)
    final_embeds = query_thought_embeddings[-1]
    main_loss, accuracy = compute_infonce_loss(
        final_embeds, pos_cand_embeds, neg_cand_embeds, temperature
    )
    
    loss_dict = {"loss_final": main_loss.item()}
    
    if K == 1:
        # Only one thought token, no auxiliary losses
        return main_loss, accuracy, loss_dict
    
    # Auxiliary losses: from intermediate thought tokens (e_1 to e_{K-1})
    aux_loss = torch.tensor(0.0, device=main_loss.device)
    
    for i in range(K - 1):
        # Weight: γ^{K-i-1} (i=0 gets γ^{K-1}, i=K-2 gets γ^1)
        weight = gamma ** (K - 1 - i)
        
        intermediate_embeds = query_thought_embeddings[i]
        step_loss, _ = compute_infonce_loss(
            intermediate_embeds, pos_cand_embeds, neg_cand_embeds, temperature
        )
        
        weighted_loss = weight * step_loss
        aux_loss = aux_loss + weighted_loss
        
        loss_dict[f"loss_thought_{i+1}"] = step_loss.item()
        loss_dict[f"weight_thought_{i+1}"] = weight
    
    # Total loss
    total_loss = main_loss + deep_supervision_lambda * aux_loss
    
    loss_dict["loss_aux_total"] = aux_loss.item()
    loss_dict["loss_total"] = total_loss.item()
    loss_dict["gamma"] = gamma
    
    return total_loss, accuracy, loss_dict


def compute_gamma(
    global_step: int,
    total_steps: int,
    gamma_start: float = 0.1,
    gamma_end: float = 0.8,
) -> float:
    """
    Compute adaptive gamma based on training progress.
    
    "让 γ 随着步数从 0.1 逐渐增加到 0.8。这就像教孩子走路：
    开始时只盯着终点（小 γ），等他会走了，再要求他每一步都走得稳（大 γ）。"
    
    Args:
        global_step: Current training step
        total_steps: Total training steps
        gamma_start: Initial gamma value (default 0.1)
        gamma_end: Final gamma value (default 0.8)
        
    Returns:
        Current gamma value
    """
    if total_steps <= 0:
        return gamma_start
    
    progress = min(global_step / total_steps, 1.0)
    gamma = gamma_start + (gamma_end - gamma_start) * progress
    
    return gamma


def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    """
    Train Jina V4 model for one epoch using contrastive loss with deep supervision.
    
    The model is a PeftModel wrapping JinaEmbeddingsV4Model. Only LoRA parameters are trained.
    Uses thought tokens and deep supervision for improved reasoning capabilities.
    
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
    metric_logger.add_meter("gamma", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0
    
    # Configuration
    task_label = getattr(config.model, 'task_label', 'retrieval')
    temperature = getattr(config.model, 'temperature', 0.07)
    max_token_length = getattr(config.model, 'max_token_length', None)
    save_steps = getattr(config.trainer_config, 'save_steps', 0)
    
    # Thought tokens and deep supervision config
    num_thought_tokens = getattr(config.model, 'num_thought_tokens', 0)
    thought_pooling_mode = getattr(config.model, 'thought_token_pooling_mode', 'last')
    symmetric_qc_encoding = bool(getattr(config.model, 'symmetric_query_candidate_encoding', False))
    cand_num_thought_tokens = num_thought_tokens if symmetric_qc_encoding else 0
    deep_supervision_lambda = getattr(config.model, 'deep_supervision_lambda', 0.1)
    gamma_start = getattr(config.model, 'gamma_start', 0.1)
    gamma_end = getattr(config.model, 'gamma_end', 0.8)
    
    # Synchronize the number of batches across all ranks to avoid NCCL deadlock
    # Different ranks may have different batch counts due to DistributedSampler + DataLoader drop_last interaction
    num_batches = len(data_loader)
    if torch.distributed.is_initialized():
        num_batches_tensor = torch.tensor([num_batches], device=gpu_id, dtype=torch.long)
        torch.distributed.all_reduce(num_batches_tensor, op=torch.distributed.ReduceOp.MIN)
        num_batches = num_batches_tensor.item()
        print(f"Synchronized batch count: {num_batches} batches per rank")
    
    # Calculate total steps for gamma scheduling
    steps_per_epoch = num_batches // accumulation_steps
    total_epochs = config.trainer_config.num_train_epochs
    total_steps = steps_per_epoch * total_epochs
    
    # Set up thought tokens in the model if needed
    if num_thought_tokens > 0:
        base_model.setup_thought_tokens(num_thought_tokens)
        print(f"Training with {num_thought_tokens} thought tokens")
        print(f"Deep supervision: lambda={deep_supervision_lambda}, gamma: {gamma_start} -> {gamma_end}")

    if symmetric_qc_encoding:
        print(
            f"[Encoding Mode] Symmetric query/candidate encoding: "
            f"both sides use num_thought_tokens={num_thought_tokens}"
        )
    else:
        print(
            f"[Encoding Mode] Asymmetric query/candidate encoding: "
            f"query uses num_thought_tokens={num_thought_tokens}, candidate uses 0"
        )
    print(f"[Thought Pooling] thought_token_pooling_mode={thought_pooling_mode}")
    
    if max_token_length is not None:
        print(f"Using max_token_length={max_token_length} for input truncation")
    
    data_iter = iter(data_loader)
    for i in metric_logger.log_every(range(num_batches), print_freq, header):
        batch = next(data_iter)
        # Move tensors appropriately
        txt_batched = batch["txt_batched"]  # RawTextBatch (stays on CPU)
        image_batched = batch["image_batched"]  # Keep on CPU
        txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
        image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)
        index_mapping = batch["index_mapping"]
        
        batch_size = len(index_mapping["query"])
        
        # Flatten indices
        query_indices = [idx[0] for idx in index_mapping["query"]]
        pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]
        
        # Handle negative candidates if present
        neg_cand_indices = []
        if "neg_cand_list" in index_mapping:
            for neg_list in index_mapping["neg_cand_list"]:
                neg_cand_indices.extend(neg_list)

        # Compute adaptive gamma
        current_gamma = compute_gamma(global_step, total_steps, gamma_start, gamma_end)

        # autocast for mixed precision
        with autocast(dtype=torch.bfloat16):
            # Encode queries with thought tokens and return all thought embeddings
            query_embeds, query_thought_embeds = encode_batch_for_training(
                model=base_model,
                txt_batched=txt_batched,
                image_batched=image_batched,
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=query_indices,
                task_label=task_label,
                device=gpu_id,
                num_thought_tokens=num_thought_tokens,
                thought_pooling_mode=thought_pooling_mode,
                return_thought_embeddings=(num_thought_tokens > 0),
                max_token_length=max_token_length,
            )
            
            # Encode positive candidates
            pos_cand_embeds, _ = encode_batch_for_training(
                model=base_model,
                txt_batched=txt_batched,
                image_batched=image_batched,
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=pos_cand_indices,
                task_label=task_label,
                device=gpu_id,
                num_thought_tokens=cand_num_thought_tokens,
                thought_pooling_mode=thought_pooling_mode,
                return_thought_embeddings=False,
                max_token_length=max_token_length,
            )
            
            # Encode negative candidates if present
            neg_cand_embeds = None
            if neg_cand_indices:
                neg_cand_embeds, _ = encode_batch_for_training(
                    model=base_model,
                    txt_batched=txt_batched,
                    image_batched=image_batched,
                    txt_mask=txt_mask,
                    image_mask=image_mask,
                    indices=neg_cand_indices,
                    task_label=task_label,
                    device=gpu_id,
                    num_thought_tokens=cand_num_thought_tokens,
                    thought_pooling_mode=thought_pooling_mode,
                    return_thought_embeddings=False,
                    max_token_length=max_token_length,
                )
            
            # Compute loss
            if num_thought_tokens > 0 and query_thought_embeds is not None:
                # Deep supervision loss
                loss, inbatch_accuracy, loss_dict = compute_deep_supervision_loss(
                    query_thought_embeddings=query_thought_embeds,
                    pos_cand_embeds=pos_cand_embeds,
                    neg_cand_embeds=neg_cand_embeds,
                    temperature=temperature,
                    gamma=current_gamma,
                    deep_supervision_lambda=deep_supervision_lambda,
                )
            else:
                # Standard contrastive loss (fallback)
                loss, inbatch_accuracy = compute_infonce_loss(
                    query_embeds=query_embeds,
                    pos_cand_embeds=pos_cand_embeds,
                    neg_cand_embeds=neg_cand_embeds,
                    temperature=temperature,
                )

        # Scale the loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulation_counter = 0
            
            # Step-level checkpoint saving
            if save_steps > 0 and global_step % save_steps == 0:
                if utils.is_main_process():
                    from models.jina_v4o.train import save_step_checkpoint
                    save_step_checkpoint(
                        model_without_ddp, optimizer, scheduler, 
                        epoch, global_step, scaler, config
                    )
                # Barrier to ensure all ranks wait for checkpoint saving
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        metric_logger.update(loss=loss.item() * accumulation_steps)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())
        metric_logger.update(gamma=current_gamma)

    # Synchronization barrier
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step


@torch.no_grad()
def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
    """
    Evaluate Jina V4 model using contrastive loss metrics.
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

    # Configuration
    task_label = getattr(config.model, 'task_label', 'retrieval')
    temperature = getattr(config.model, 'temperature', 0.07)
    num_thought_tokens = getattr(config.model, 'num_thought_tokens', 0)
    thought_pooling_mode = getattr(config.model, 'thought_token_pooling_mode', 'last')
    symmetric_qc_encoding = bool(getattr(config.model, 'symmetric_query_candidate_encoding', False))
    cand_num_thought_tokens = num_thought_tokens if symmetric_qc_encoding else 0
    
    # Set up thought tokens for evaluation
    if num_thought_tokens > 0:
        base_model.setup_thought_tokens(num_thought_tokens)
        print(f"Evaluating with {num_thought_tokens} thought tokens")

    if symmetric_qc_encoding:
        print(
            f"Evaluating with symmetric query/candidate encoding: "
            f"both sides use num_thought_tokens={num_thought_tokens}"
        )
    else:
        print(
            f"Evaluating with asymmetric query/candidate encoding: "
            f"query uses num_thought_tokens={num_thought_tokens}, candidate uses 0"
        )
    print(f"Evaluating with thought_token_pooling_mode={thought_pooling_mode}")

    # Synchronize the number of batches across all ranks to avoid NCCL deadlock
    num_batches = len(data_loader)
    if torch.distributed.is_initialized():
        num_batches_tensor = torch.tensor([num_batches], device=gpu_id, dtype=torch.long)
        torch.distributed.all_reduce(num_batches_tensor, op=torch.distributed.ReduceOp.MIN)
        num_batches = num_batches_tensor.item()

    data_iter = iter(data_loader)
    for i in metric_logger.log_every(range(num_batches), print_freq, header):
        batch = next(data_iter)
        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
        image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)
        index_mapping = batch["index_mapping"]
        
        query_indices = [idx[0] for idx in index_mapping["query"]]
        pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]

        with autocast():
            # Encode queries
            query_embeds, _ = encode_batch_for_training(
                model=base_model,
                txt_batched=txt_batched,
                image_batched=image_batched,
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=query_indices,
                task_label=task_label,
                device=gpu_id,
                num_thought_tokens=num_thought_tokens,
                thought_pooling_mode=thought_pooling_mode,
                return_thought_embeddings=False,
            )
            
            # Encode positive candidates
            pos_cand_embeds, _ = encode_batch_for_training(
                model=base_model,
                txt_batched=txt_batched,
                image_batched=image_batched,
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=pos_cand_indices,
                task_label=task_label,
                device=gpu_id,
                num_thought_tokens=cand_num_thought_tokens,
                thought_pooling_mode=thought_pooling_mode,
                return_thought_embeddings=False,
            )
            
            # Compute loss
            loss, inbatch_accuracy = compute_infonce_loss(
                query_embeds=query_embeds,
                pos_cand_embeds=pos_cand_embeds,
                neg_cand_embeds=None,
                temperature=temperature,
            )

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # Synchronization
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
