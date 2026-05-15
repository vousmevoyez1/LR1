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
import torch.distributed as dist
from torch.cuda.amp import autocast
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

import models.jina_v4oauxa.utils as utils


def _all_gather_with_local_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-gather tensor across ranks while preserving gradient for local slice.

    This is used for cross-device in-batch negatives in contrastive learning.
    Remote slices are gathered as detached tensors; local slice is replaced by
    the original tensor so gradients can flow to local parameters.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.detach())
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


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
    cross_device_negatives: bool = False,
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
    
    use_cross_device = cross_device_negatives and dist.is_available() and dist.is_initialized()

    # Build positive candidate bank (local or global via all-gather)
    if use_cross_device:
        pos_bank = _all_gather_with_local_grad(pos_cand_embeds)
        local_batch_size = pos_cand_embeds.size(0)
        labels = torch.arange(batch_size, device=device) + dist.get_rank() * local_batch_size
    else:
        pos_bank = pos_cand_embeds
        labels = torch.arange(batch_size, device=device)

    # Similarity matrix: [batch_size, bank_size]
    sim_matrix = torch.matmul(query_embeds, pos_bank.t()) / temperature
    
    # Add hard negatives if available
    if neg_cand_embeds is not None and neg_cand_embeds.size(0) > 0:
        if use_cross_device:
            neg_cand_embeds = _all_gather_with_local_grad(neg_cand_embeds)
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
    progressive_constraint_lambda: float = 10.0,
    progressive_constraint_margin: float = 0.05,
    enable_directional_alignment_loss: bool = False,
    directional_alignment_lambda: float = 0.0,
    directional_alignment_eps: float = 1e-8,
    cross_device_negatives: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
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
        Tuple of:
            - total_loss
            - in-batch accuracy
            - loss_dict (float scalars for logging)
            - loss_components (tensor losses for debug gradient analysis)
    """
    def compute_directional_alignment_loss(
        thought_embeddings: List[torch.Tensor],
        eps: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Encourage consistent incremental direction across adjacent thought steps.

        Let Δ_i = e_{i+1} - e_i. We minimize:
            L_dir = mean_i,batch (1 - cos(Δ_i, Δ_{i+1}))
        which enforces local tangent continuity along the thought trajectory.
        """
        K_local = len(thought_embeddings)
        if K_local < 3:
            zero_loss = thought_embeddings[0].new_zeros(())
            return zero_loss, {
                "direction_alignment_pairs": 0.0,
                "direction_alignment_cosine_mean": 1.0,
            }

        deltas = [thought_embeddings[i + 1] - thought_embeddings[i] for i in range(K_local - 1)]
        deltas = [F.normalize(delta, p=2, dim=-1, eps=eps) for delta in deltas]

        pairwise_cosines = []
        for i in range(len(deltas) - 1):
            cosine_i = (deltas[i] * deltas[i + 1]).sum(dim=-1)  # [batch]
            pairwise_cosines.append(cosine_i)

        cosine_tensor = torch.stack(pairwise_cosines, dim=0)  # [K-2, batch]
        direction_loss = (1.0 - cosine_tensor).mean()

        return direction_loss, {
            "direction_alignment_pairs": float(len(pairwise_cosines)),
            "direction_alignment_cosine_mean": cosine_tensor.mean().item(),
        }

    K = len(query_thought_embeddings)
    
    if K == 0:
        # Fallback: no thought tokens, shouldn't happen
        raise ValueError("No thought embeddings provided for deep supervision")
    
    # Main loss: from the last thought token (e_K)
    final_embeds = query_thought_embeddings[-1]
    main_loss, accuracy = compute_infonce_loss(
        final_embeds,
        pos_cand_embeds,
        neg_cand_embeds,
        temperature,
        cross_device_negatives=cross_device_negatives,
    )
    
    loss_dict = {"loss_final": main_loss.item()}
    
    if K == 1:
        # Only one thought token, no auxiliary losses
        loss_dict["loss_aux_total"] = 0.0
        loss_dict["loss_progressive_constraint"] = 0.0
        loss_dict["loss_progressive_constraint_weighted"] = 0.0
        loss_dict["progressive_violation_rate"] = 0.0
        loss_dict["loss_direction_alignment"] = 0.0
        loss_dict["loss_direction_alignment_weighted"] = 0.0
        loss_dict["loss_total"] = main_loss.item()
        loss_dict["gamma"] = gamma
        loss_dict["direction_alignment_pairs"] = 0.0
        loss_dict["direction_alignment_cosine_mean"] = 1.0
        return main_loss, accuracy, loss_dict, {
            "main_loss": main_loss,
            "deep_aux_loss_unweighted": main_loss.new_zeros(()),
            "deep_aux_loss_weighted": main_loss.new_zeros(()),
            "progressive_loss_unweighted": main_loss.new_zeros(()),
            "progressive_loss_weighted": main_loss.new_zeros(()),
            "directional_aux_loss_unweighted": main_loss.new_zeros(()),
            "directional_aux_loss_weighted": main_loss.new_zeros(()),
        }
    
    # Auxiliary losses: from intermediate thought tokens (e_1 to e_{K-1})
    aux_loss = torch.tensor(0.0, device=main_loss.device)
    step_losses = []
    
    for i in range(K - 1):
        # Weight: γ^{K-i-1} (i=0 gets γ^{K-1}, i=K-2 gets γ^1)
        weight = gamma ** (K - 1 - i)
        
        intermediate_embeds = query_thought_embeddings[i]
        step_loss, _ = compute_infonce_loss(
            intermediate_embeds,
            pos_cand_embeds,
            neg_cand_embeds,
            temperature,
            cross_device_negatives=cross_device_negatives,
        )
        
        weighted_loss = weight * step_loss
        aux_loss = aux_loss + weighted_loss
        step_losses.append(step_loss)
        
        loss_dict[f"loss_thought_{i+1}"] = step_loss.item()
        loss_dict[f"weight_thought_{i+1}"] = weight

    # Append final step loss so we have [L1, L2, ..., LK]
    step_losses.append(main_loss)
    for i, step_loss in enumerate(step_losses):
        loss_dict[f"loss_step_{i+1}"] = step_loss.item()

    # Strong progressive monotonic constraint:
    # enforce L_{i+1} <= L_i - margin for all adjacent steps
    # and use a quadratic penalty so larger violations are punished more heavily.
    progressive_violations = []
    for i in range(K - 1):
        violation_i = F.relu(step_losses[i + 1] - step_losses[i] + progressive_constraint_margin)
        progressive_violations.append(violation_i.pow(2))
        loss_dict[f"progressive_violation_{i+1}_to_{i+2}"] = violation_i.item()

    if len(progressive_violations) > 0:
        progressive_violation_tensor = torch.stack(progressive_violations)
        progressive_loss = progressive_violation_tensor.mean()
        progressive_violation_rate = (progressive_violation_tensor > 0).float().mean().item()
    else:
        progressive_loss = main_loss.new_zeros(())
        progressive_violation_rate = 0.0
    
    # Directional alignment auxiliary loss (kept disabled by default).
    if enable_directional_alignment_loss and directional_alignment_lambda > 0:
        direction_loss, direction_stats = compute_directional_alignment_loss(
            query_thought_embeddings, directional_alignment_eps
        )
    else:
        direction_loss = main_loss.new_zeros(())
        direction_stats = {
            "direction_alignment_pairs": 0.0,
            "direction_alignment_cosine_mean": 1.0,
        }

    weighted_deep_aux = deep_supervision_lambda * aux_loss
    weighted_progressive = progressive_constraint_lambda * progressive_loss
    weighted_direction_aux = directional_alignment_lambda * direction_loss

    # Total loss
    total_loss = main_loss + weighted_deep_aux + weighted_progressive + weighted_direction_aux
    
    loss_dict["loss_aux_total"] = aux_loss.item()
    loss_dict["loss_progressive_constraint"] = progressive_loss.item()
    loss_dict["loss_progressive_constraint_weighted"] = weighted_progressive.item()
    loss_dict["progressive_violation_rate"] = progressive_violation_rate
    loss_dict["loss_direction_alignment"] = direction_loss.item()
    loss_dict["loss_direction_alignment_weighted"] = weighted_direction_aux.item()
    loss_dict["loss_total"] = total_loss.item()
    loss_dict["gamma"] = gamma
    loss_dict["direction_alignment_pairs"] = direction_stats["direction_alignment_pairs"]
    loss_dict["direction_alignment_cosine_mean"] = direction_stats["direction_alignment_cosine_mean"]
    
    return total_loss, accuracy, loss_dict, {
        "main_loss": main_loss,
        "deep_aux_loss_unweighted": aux_loss,
        "deep_aux_loss_weighted": weighted_deep_aux,
        "progressive_loss_unweighted": progressive_loss,
        "progressive_loss_weighted": weighted_progressive,
        "directional_aux_loss_unweighted": direction_loss,
        "directional_aux_loss_weighted": weighted_direction_aux,
    }


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


def compute_hard_gate_stage2_loss(
    base_model,
    query_thought_embeddings: List[torch.Tensor],
    query_context_embeds: torch.Tensor,
    pos_cand_embeds: torch.Tensor,
    scale: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
        Stage-2: Gumbel-Softmax 路由 + In-Batch InfoNCE 损失。

        输入：
            - query_thought_embeddings: 长度 K 的 List，每个元素 [B, D]
            - query_context_embeds: [B, D]，主模型编码后的 query 部分 embedding
            - pos_cand_embeds: [B, D]

        流程：
            1) thought_tokens = stack(...) -> [B, K, D]
            2) Gate 使用融合(query_context_embeds, thought_tokens[:,0,:])预测步骤 logits [B, K]
            3) 训练时 select_best_thought_embedding 内部使用 gumbel_softmax(hard=True)
               得到 selected_query_embeddings [B, D]
            4) sim_matrix = query @ pos^T -> [B, B]
            5) targets = arange(B)
            6) loss = CrossEntropy(sim_matrix * scale, targets)

        说明：
            - 监督目标保持不变（teacher-best step）。
            - teacher 相似度仅用于监控统计，不参与反传。
    """
    if len(query_thought_embeddings) == 0:
        raise ValueError("Stage-2 hard-gate training requires non-empty query_thought_embeddings")

    # thought_tokens: [B, K, D]
    thought_tokens = torch.stack(query_thought_embeddings, dim=1)

    # 由 Gate + (train时)Gumbel-Softmax 选出的 query 表征: [B, D]
    # step_logits: [B, K], selected_steps: [B]
    query_embeds, step_logits, selected_steps = base_model.select_best_thought_embedding(
        thought_tokens,
        query_context_embeddings=query_context_embeds,
    )

    # 归一化后计算 in-batch 相似度矩阵
    # query_embeds: [B, D], pos_norm: [B, D]
    query_norm = F.normalize(query_embeds, p=2, dim=-1)
    pos_norm = F.normalize(pos_cand_embeds, p=2, dim=-1)

    # sim_matrix: [B, B]，第 i 行与 batch 内所有正样本比较
    sim_matrix = torch.matmul(query_norm, pos_norm.t())
    sim_matrix = sim_matrix * float(scale)

    # 目标标签：对角线为正样本
    targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    # InfoNCE / in-batch retrieval loss
    loss = F.cross_entropy(sim_matrix, targets)

    # retrieval in-batch accuracy
    retrieval_pred = sim_matrix.argmax(dim=-1)
    inbatch_accuracy = (retrieval_pred == targets).float().mean()

    # teacher 相似度（仅监控，不反传）: [B, K]
    thought_tokens_norm = F.normalize(thought_tokens, p=2, dim=-1)
    similarities = (thought_tokens_norm * pos_norm.unsqueeze(1)).sum(dim=-1).detach()
    teacher_best_steps = similarities.argmax(dim=-1)

    # gate step 分类准确率（与 teacher-best 对齐程度）
    gate_acc = (selected_steps == teacher_best_steps).float().mean()

    # 监控 selected step 对应 teacher cosine
    target_cosine = similarities.gather(1, teacher_best_steps.unsqueeze(1)).mean()
    selected_cosine = similarities.gather(1, selected_steps.unsqueeze(1)).mean()

    stats = {
        "gate_target_cosine": target_cosine.item(),
        "gate_selected_cosine": selected_cosine.item(),
        "gate_label_accuracy": gate_acc.item(),
        "stage2_inbatch_accuracy": inbatch_accuracy.item(),
    }
    return loss, inbatch_accuracy, stats


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
    metric_logger.add_meter("loss_final", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_aux_total", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_progressive_constraint", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_progressive_constraint_weighted", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("progressive_violation_rate", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_direction_alignment", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_direction_alignment_weighted", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("grad_ratio_aux_over_main", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("grad_norm_main", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("grad_norm_aux", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("gamma", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("gate_target_cosine", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("gate_selected_cosine", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("gate_label_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
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
    progressive_constraint_lambda = getattr(config.model, 'progressive_constraint_lambda', 10.0)
    progressive_constraint_margin = getattr(config.model, 'progressive_constraint_margin', 0.05)
    enable_directional_alignment_loss = bool(getattr(config.model, 'enable_directional_alignment_loss', False))
    directional_alignment_lambda = getattr(config.model, 'directional_alignment_lambda', 0.0)
    directional_alignment_eps = getattr(config.model, 'directional_alignment_eps', 1e-8)
    cross_device_negatives = bool(getattr(config.model, 'cross_device_negatives', True))
    debug_aux_loss = bool(getattr(config.model, 'debug_aux_loss', False))
    debug_aux_loss_print_freq = int(getattr(config.model, 'debug_aux_loss_print_freq', print_freq))
    debug_train = bool(getattr(config.model, 'debug_train', True))
    debug_train_print_freq = int(getattr(config.model, 'debug_train_print_freq', print_freq))
    debug_batch_composition = bool(getattr(config.model, 'debug_batch_composition', True))
    debug_embedding_stats = bool(getattr(config.model, 'debug_embedding_stats', True))
    debug_progressive_detail = bool(getattr(config.model, 'debug_progressive_detail', True))
    gamma_start = getattr(config.model, 'gamma_start', 0.1)
    gamma_end = getattr(config.model, 'gamma_end', 0.8)
    training_stage = int(getattr(config.model, 'training_stage', 1))
    hard_gate_mlp_hidden_dim = getattr(config.model, 'hard_gate_mlp_hidden_dim', None)
    hard_gate_dropout = float(getattr(config.model, 'hard_gate_dropout', 0.0))
    hard_gate_infonce_scale = float(getattr(config.model, 'hard_gate_infonce_scale', 20.0))
    stage2_query_context_pooling = getattr(config.model, 'stage2_query_context_pooling', 'last')
    if stage2_query_context_pooling == 'hard_gate':
        # Avoid recursive dependency: gate input should come from base-model query context, not gate output.
        stage2_query_context_pooling = 'last'
    
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
        print(
            f"Deep supervision: lambda={deep_supervision_lambda}, gamma: {gamma_start} -> {gamma_end}; "
            f"progressive_constraint_lambda={progressive_constraint_lambda}, "
            f"progressive_constraint_margin={progressive_constraint_margin}, "
            f"directional_alignment_enabled={enable_directional_alignment_loss}, "
            f"directional_alignment_lambda={directional_alignment_lambda}"
        )
        if debug_aux_loss:
            print(
                f"[Debug] Auxiliary loss debugging enabled. "
                f"Print frequency={debug_aux_loss_print_freq}"
            )
        if debug_train:
            print(
                f"[Debug] Train debug enabled. "
                f"freq={debug_train_print_freq}, "
                f"batch_comp={debug_batch_composition}, "
                f"embed_stats={debug_embedding_stats}, "
                f"progressive_detail={debug_progressive_detail}"
            )

    if training_stage == 2:
        base_model.setup_hard_gate(
            mlp_hidden_dim=hard_gate_mlp_hidden_dim,
            dropout=hard_gate_dropout,
        )
        print("[Stage-2] Hard-gate training enabled (joint finetune mode)")

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
    print(f"[Negatives] cross_device_negatives={cross_device_negatives}")
    
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

        if (
            debug_train
            and debug_batch_composition
            and (i % max(debug_train_print_freq, 1) == 0)
            and utils.is_main_process()
        ):
            q_txt_ratio = txt_mask[query_indices].float().mean().item() if len(query_indices) > 0 else 0.0
            q_img_ratio = image_mask[query_indices].float().mean().item() if len(query_indices) > 0 else 0.0
            p_txt_ratio = txt_mask[pos_cand_indices].float().mean().item() if len(pos_cand_indices) > 0 else 0.0
            p_img_ratio = image_mask[pos_cand_indices].float().mean().item() if len(pos_cand_indices) > 0 else 0.0
            neg_per_query = (len(neg_cand_indices) / max(batch_size, 1)) if len(neg_cand_indices) > 0 else 0.0
            print(
                "[Debug][Batch] "
                f"epoch={epoch} iter={i} step={global_step} "
                f"bs={batch_size} q={len(query_indices)} p={len(pos_cand_indices)} "
                f"neg={len(neg_cand_indices)} neg_per_q={neg_per_query:.2f} "
                f"q_txt_ratio={q_txt_ratio:.3f} q_img_ratio={q_img_ratio:.3f} "
                f"p_txt_ratio={p_txt_ratio:.3f} p_img_ratio={p_img_ratio:.3f}"
            )

        # Compute adaptive gamma
        current_gamma = compute_gamma(global_step, total_steps, gamma_start, gamma_end)

        loss_dict = {}

        # autocast for mixed precision
        with autocast(dtype=torch.bfloat16):
            if training_stage == 2:
                if num_thought_tokens <= 0:
                    raise ValueError("Stage-2 hard-gate training requires model.num_thought_tokens > 0")

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
                    thought_pooling_mode=stage2_query_context_pooling,
                    return_thought_embeddings=True,
                    max_token_length=max_token_length,
                )

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

                if query_thought_embeds is None:
                    raise RuntimeError("Stage-2 expected query thought embeddings, but got None")

                loss, inbatch_accuracy, stage2_stats = compute_hard_gate_stage2_loss(
                    base_model=base_model,
                    query_thought_embeddings=query_thought_embeds,
                    query_context_embeds=query_embeds,
                    pos_cand_embeds=pos_cand_embeds,
                    scale=hard_gate_infonce_scale,
                )

                loss_dict.update(stage2_stats)
            else:
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
                    loss, inbatch_accuracy, loss_dict, loss_components = compute_deep_supervision_loss(
                        query_thought_embeddings=query_thought_embeds,
                        pos_cand_embeds=pos_cand_embeds,
                        neg_cand_embeds=neg_cand_embeds,
                        temperature=temperature,
                        gamma=current_gamma,
                        deep_supervision_lambda=deep_supervision_lambda,
                        progressive_constraint_lambda=progressive_constraint_lambda,
                        progressive_constraint_margin=progressive_constraint_margin,
                        enable_directional_alignment_loss=enable_directional_alignment_loss,
                        directional_alignment_lambda=directional_alignment_lambda,
                        directional_alignment_eps=directional_alignment_eps,
                        cross_device_negatives=cross_device_negatives,
                    )

                    # Optional debug: compare gradient magnitude between main loss and aux (directional) loss
                    # 【关键修复】禁用梯度调试以节省显存 - 这会创建大量中间张量
                    if debug_aux_loss and False:  # 强制禁用以避免OOM
                        def _grad_norm(grads: List[Optional[torch.Tensor]]) -> float:
                            total_sq = 0.0
                            for g in grads:
                                if g is None:
                                    continue
                                total_sq += float(torch.sum((g.float()) ** 2).item())
                            return total_sq ** 0.5

                        main_grads = torch.autograd.grad(
                            loss_components["main_loss"],
                            query_thought_embeds,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        aux_grads = torch.autograd.grad(
                            loss_components["progressive_loss_weighted"],
                            query_thought_embeds,
                            retain_graph=True,
                            allow_unused=True,
                        )

                        grad_norm_main = _grad_norm(list(main_grads))
                        grad_norm_aux = _grad_norm(list(aux_grads))
                        grad_ratio_aux_over_main = grad_norm_aux / max(grad_norm_main, 1e-12)

                        loss_dict["grad_norm_main"] = grad_norm_main
                        loss_dict["grad_norm_aux"] = grad_norm_aux
                        loss_dict["grad_ratio_aux_over_main"] = grad_ratio_aux_over_main

                        if (i % max(debug_aux_loss_print_freq, 1) == 0) and utils.is_main_process():
                            print(
                                "[Debug][AuxLoss] "
                                f"step={global_step} "
                                f"loss_main={loss_dict['loss_final']:.6f} "
                                f"loss_prog={loss_dict.get('loss_progressive_constraint', 0.0):.6f} "
                                f"loss_prog_w={loss_dict.get('loss_progressive_constraint_weighted', 0.0):.6f} "
                                f"prog_violation_rate={loss_dict.get('progressive_violation_rate', 0.0):.4f} "
                                f"loss_aux_dir={loss_dict['loss_direction_alignment']:.6f} "
                                f"loss_aux_dir_weighted={loss_dict['loss_direction_alignment_weighted']:.6f} "
                                f"grad_norm_main={grad_norm_main:.6e} "
                                f"grad_norm_aux={grad_norm_aux:.6e} "
                                f"grad_ratio_aux_over_main={grad_ratio_aux_over_main:.6f}"
                            )
                    else:
                        # 设置默认值避免后续代码报错
                        loss_dict["grad_norm_main"] = 0.0
                        loss_dict["grad_norm_aux"] = 0.0
                        loss_dict["grad_ratio_aux_over_main"] = 0.0
                else:
                    # Standard contrastive loss (fallback)
                    loss, inbatch_accuracy = compute_infonce_loss(
                        query_embeds=query_embeds,
                        pos_cand_embeds=pos_cand_embeds,
                        neg_cand_embeds=neg_cand_embeds,
                        temperature=temperature,
                        cross_device_negatives=cross_device_negatives,
                    )

        if (
            debug_train
            and debug_embedding_stats
            and (i % max(debug_train_print_freq, 1) == 0)
            and utils.is_main_process()
        ):
            query_norm_mean = query_embeds.detach().norm(p=2, dim=-1).mean().item()
            pos_norm_mean = pos_cand_embeds.detach().norm(p=2, dim=-1).mean().item()
            pos_cos_mean = F.cosine_similarity(
                query_embeds.detach(),
                pos_cand_embeds.detach(),
                dim=-1,
            ).mean().item()

            debug_msg = (
                "[Debug][Embed] "
                f"epoch={epoch} iter={i} step={global_step} "
                f"q_norm={query_norm_mean:.4f} p_norm={pos_norm_mean:.4f} "
                f"q_pos_cos={pos_cos_mean:.4f} "
                f"loss={loss.item():.6f}"
            )

            if (
                num_thought_tokens > 0
                and query_thought_embeds is not None
                and debug_progressive_detail
                and training_stage != 2
            ):
                step_cos_list = [
                    F.cosine_similarity(step_emb.detach(), pos_cand_embeds.detach(), dim=-1).mean().item()
                    for step_emb in query_thought_embeds
                ]
                step_loss_list = [
                    loss_dict.get(f"loss_step_{k+1}", float("nan"))
                    for k in range(len(query_thought_embeds))
                ]
                progressive_cos_improve_rate = 0.0
                if len(step_cos_list) > 1:
                    improve_flags = [
                        1.0 if step_cos_list[k + 1] > step_cos_list[k] else 0.0
                        for k in range(len(step_cos_list) - 1)
                    ]
                    progressive_cos_improve_rate = sum(improve_flags) / len(improve_flags)

                debug_msg += (
                    f" step_loss={['%.4f' % x for x in step_loss_list]}"
                    f" step_cos={['%.4f' % x for x in step_cos_list]}"
                    f" cos_improve_rate={progressive_cos_improve_rate:.3f}"
                )

            print(debug_msg)

        # Scale the loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        # Note: bfloat16 doesn't need GradScaler (it has better numerical stability than float16)
        # GradScaler only works with float16, not bfloat16
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
                    from models.jina_v4oauxa.train import save_step_checkpoint
                    save_step_checkpoint(
                        model_without_ddp, optimizer, scheduler,
                        epoch, global_step, scaler, config
                    )
                # Barrier to ensure all ranks wait for checkpoint saving
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        metric_logger.update(loss=loss.item() * accumulation_steps)
        if num_thought_tokens > 0 and query_thought_embeds is not None:
            if "loss_final" in loss_dict:
                metric_logger.update(loss_final=loss_dict["loss_final"])
            if "loss_aux_total" in loss_dict:
                metric_logger.update(loss_aux_total=loss_dict["loss_aux_total"])
            if "loss_progressive_constraint" in loss_dict:
                metric_logger.update(loss_progressive_constraint=loss_dict["loss_progressive_constraint"])
            if "loss_progressive_constraint_weighted" in loss_dict:
                metric_logger.update(loss_progressive_constraint_weighted=loss_dict["loss_progressive_constraint_weighted"])
            if "progressive_violation_rate" in loss_dict:
                metric_logger.update(progressive_violation_rate=loss_dict["progressive_violation_rate"])
            if "loss_direction_alignment" in loss_dict:
                metric_logger.update(loss_direction_alignment=loss_dict["loss_direction_alignment"])
            if "loss_direction_alignment_weighted" in loss_dict:
                metric_logger.update(loss_direction_alignment_weighted=loss_dict["loss_direction_alignment_weighted"])
            if "grad_norm_main" in loss_dict:
                metric_logger.update(grad_norm_main=loss_dict["grad_norm_main"])
            if "grad_norm_aux" in loss_dict:
                metric_logger.update(grad_norm_aux=loss_dict["grad_norm_aux"])
            if "grad_ratio_aux_over_main" in loss_dict:
                metric_logger.update(grad_ratio_aux_over_main=loss_dict["grad_ratio_aux_over_main"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())
        if training_stage == 2:
            if "gate_target_cosine" in loss_dict:
                metric_logger.update(gate_target_cosine=loss_dict["gate_target_cosine"])
            if "gate_selected_cosine" in loss_dict:
                metric_logger.update(gate_selected_cosine=loss_dict["gate_selected_cosine"])
            if "gate_label_accuracy" in loss_dict:
                metric_logger.update(gate_label_accuracy=loss_dict["gate_label_accuracy"])
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
    cross_device_negatives = bool(getattr(config.model, 'cross_device_negatives', True))
    training_stage = int(getattr(config.model, 'training_stage', 1))
    hard_gate_mlp_hidden_dim = getattr(config.model, 'hard_gate_mlp_hidden_dim', None)
    hard_gate_dropout = float(getattr(config.model, 'hard_gate_dropout', 0.0))
    hard_gate_infonce_scale = float(getattr(config.model, 'hard_gate_infonce_scale', 20.0))
    stage2_query_context_pooling = getattr(config.model, 'stage2_query_context_pooling', 'last')
    if stage2_query_context_pooling == 'hard_gate':
        stage2_query_context_pooling = 'last'
    
    # Set up thought tokens for evaluation
    if num_thought_tokens > 0:
        base_model.setup_thought_tokens(num_thought_tokens)
        print(f"Evaluating with {num_thought_tokens} thought tokens")

    if training_stage == 2:
        base_model.setup_hard_gate(
            mlp_hidden_dim=hard_gate_mlp_hidden_dim,
            dropout=hard_gate_dropout,
        )
        print("[Stage-2] Hard-gate evaluation enabled")

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
    print(f"Evaluating with cross_device_negatives={cross_device_negatives}")

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
            if training_stage == 2:
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
                    thought_pooling_mode=stage2_query_context_pooling,
                    return_thought_embeddings=True,
                )

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

                if query_thought_embeds is None:
                    raise RuntimeError("Stage-2 evaluation expected query thought embeddings, but got None")

                loss, inbatch_accuracy, _ = compute_hard_gate_stage2_loss(
                    base_model=base_model,
                    query_thought_embeddings=query_thought_embeds,
                    query_context_embeds=query_embeds,
                    pos_cand_embeds=pos_cand_embeds,
                    scale=hard_gate_infonce_scale,
                )
            else:
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
                    cross_device_negatives=cross_device_negatives,
                )

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # Synchronization
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
