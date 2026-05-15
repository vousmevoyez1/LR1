import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional, Union

from models.qwen3a import utils


def _debug_enabled(config) -> bool:
    return bool(
        getattr(config.trainer_config, "debug_mode", False)
        or getattr(config.model, "debug_mode", False)
    )


def _debug_log(config, message: str):
    if _debug_enabled(config) and utils.is_main_process():
        print(f"[DEBUG] {message}")


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
    use_final_token: bool = False,
    reason_steps: int = 0,
    use_cache_for_reasoning: bool = False,
    #max_token_length: Optional[int] = None,
    max_token_length=15000,       # [新增] 接收截断参数
    max_visual_pixels=501760,
    return_reasoning_steps: bool = False,
) -> Union[torch.Tensor, Any]:
    """
    Encode a subset of batch items using Qwen3-VL with optional thought tokens.

    This function expects a model wrapper that provides:
      - model._preprocess_inputs(texts=..., images=..., num_thought_tokens=...)
      - model._forward_embeddings(**inputs, num_thought_tokens=...)

    Notes:
    - Asymmetric encoding is handled by the caller via num_thought_tokens:
        * queries: num_thought_tokens=K
        * candidates: num_thought_tokens=0
            and use_final_token:
                * queries: use_final_token follows config
                * candidates: use_final_token=False (always native pooling)
    - reason_steps / use_cache_for_reasoning are kept for compatibility but not used.

    Args:
        model: Qwen3VLThoughtWrapper instance
        txt_batched: RawTextBatch containing raw strings
        image_batched: Image tensors [total_batch, 3, H, W] in [0, 1] range (SHOULD BE ON CPU)
        txt_mask: Text presence mask [total_batch] (SHOULD BE ON GPU)
        image_mask: Image presence mask [total_batch] (SHOULD BE ON GPU)
        indices: Indices of items to encode from the flattened batch
        task_label: Task identifier (kept for compatibility)
        device: Target device
        num_thought_tokens: Number of thought tokens for query encoding (0 disables thought tokens)
        use_final_token: Whether to append/use <final> token for pooling in this call
        max_token_length: Maximum token length for truncation (None = use default)

    Returns:
        - If return_reasoning_steps=False: Tensor of embeddings [len(indices), embed_dim]
        - If return_reasoning_steps=True: Qwen3ThoughtOutput(single_vec_emb, thought_embeddings)
    """
    from models.qwen3a.qwen3_thought_wrapper import Qwen3ThoughtOutput

    # 1. [新增] 提取出剥离了 DDP 外壳的基础模型，专门用来调用预处理函数
    unwrapped_model = model.module if hasattr(model, "module") else model
    if len(indices) == 0:
        embed_dim = unwrapped_model.model.config.text_config.hidden_size
        empty = torch.zeros(0, embed_dim, device=device)
        if return_reasoning_steps:
            return Qwen3ThoughtOutput(single_vec_emb=empty, thought_embeddings=None)
        return empty

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

    embed_dim = unwrapped_model.model.config.text_config.hidden_size
    model_dtype = next(unwrapped_model.parameters()).dtype
    embeddings = torch.zeros(len(indices), embed_dim, device=device, dtype=model_dtype)
    reasoning_step_embeddings = None

    def process_modality_group(texts=None, pils=None):
        # 预处理（已在 train_one_epoch 做了全 batch 的 overlong 丢弃，这里不再做长度检查）
        inputs = unwrapped_model._preprocess_inputs(
            texts=texts,
            images=pils,
            num_thought_tokens=num_thought_tokens,
            use_final_token=use_final_token,
            max_token_length=max_token_length,
            max_visual_pixels=max_visual_pixels,
        )

        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        output = model(
            encode_mbeir_batch=False,
            num_thought_tokens=num_thought_tokens,
            use_final_token=use_final_token,
            **inputs,
        )
        return output
    # Process text-only samples
    if text_only_idx:
        local_indices = [x[0] for x in text_only_idx]
        global_indices = [x[1] for x in text_only_idx]
        texts = txt_batched[global_indices]

        # inputs = unwrapped_model._preprocess_inputs(texts=texts, images=None, num_thought_tokens=num_thought_tokens)
        # inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        # output = model(encode_mbeir_batch=False, num_thought_tokens=num_thought_tokens, **inputs)
        # embeddings[local_indices] = output.single_vec_emb.to(model_dtype)
        output = process_modality_group(texts=texts)
        embeddings[local_indices] = output.single_vec_emb.to(model_dtype)
        if return_reasoning_steps and output.thought_embeddings is not None:
            if reasoning_step_embeddings is None:
                reasoning_step_embeddings = [
                    torch.zeros(len(indices), embed_dim, device=device, dtype=model_dtype)
                    for _ in output.thought_embeddings
                ]
            for step_idx, step_emb in enumerate(output.thought_embeddings):
                reasoning_step_embeddings[step_idx][local_indices] = step_emb.to(model_dtype)

    

    # 3. 修改 Image-only 处理段
    if image_only_idx:
        local_indices = [x[0] for x in image_only_idx]
        global_indices = [x[1] for x in image_only_idx]

        pil_images = [to_pil_image(image_batched[idx]) for idx in global_indices]
        empty_texts = [""] * len(pil_images)
        output = process_modality_group(texts=empty_texts, pils=pil_images)
        embeddings[local_indices] = output.single_vec_emb.to(model_dtype)
        if return_reasoning_steps and output.thought_embeddings is not None:
            if reasoning_step_embeddings is None:
                reasoning_step_embeddings = [
                    torch.zeros(len(indices), embed_dim, device=device, dtype=model_dtype)
                    for _ in output.thought_embeddings
                ]
            for step_idx, step_emb in enumerate(output.thought_embeddings):
                reasoning_step_embeddings[step_idx][local_indices] = step_emb.to(model_dtype)


    # 4. 修改 Multimodal 处理段
    if multimodal_idx:
        local_indices = [x[0] for x in multimodal_idx]
        global_indices = [x[1] for x in multimodal_idx]

        pil_images = [to_pil_image(image_batched[idx]) for idx in global_indices]
        texts = txt_batched[global_indices]
        output = process_modality_group(texts=texts, pils=pil_images)
        embeddings[local_indices] = output.single_vec_emb.to(model_dtype)
        if return_reasoning_steps and output.thought_embeddings is not None:
            if reasoning_step_embeddings is None:
                reasoning_step_embeddings = [
                    torch.zeros(len(indices), embed_dim, device=device, dtype=model_dtype)
                    for _ in output.thought_embeddings
                ]
            for step_idx, step_emb in enumerate(output.thought_embeddings):
                reasoning_step_embeddings[step_idx][local_indices] = step_emb.to(model_dtype)


    if return_reasoning_steps:
        return Qwen3ThoughtOutput(
            single_vec_emb=embeddings,
            thought_embeddings=reasoning_step_embeddings,
        )

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


def compute_reasoning_progressive_regularizer(
    step_losses: List[torch.Tensor],
    margin: float,
) -> torch.Tensor:
    """
    Progressive hinge-squared penalty:
    L_prog = 1/(K-1) * sum_i [max(0, L_{i+1} - L_i + m)]^2
    where the last step is the final token loss.
    """
    if len(step_losses) <= 1:
        return step_losses[0].new_zeros(()) if step_losses else torch.tensor(0.0)

    penalties = []
    for i in range(len(step_losses) - 1):
        diff = step_losses[i + 1] - step_losses[i] + margin
        penalties.append(F.relu(diff) ** 2)
    return torch.stack(penalties).mean()


def compute_multistep_reasoning_loss(
    query_step_embeddings: Optional[List[torch.Tensor]],
    pos_cand_embeds: torch.Tensor,
    neg_cand_embeds: Optional[torch.Tensor],
    temperature: float,
    min_step_weight: float = 0.4,
    max_step_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Progressive multi-step InfoNCE supervision:
    L_total = sum_{k=0}^{K} w_k * L_InfoNCE^{(k)}
    where w_0=min_step_weight, w_K=max_step_weight, and intermediate w_k are
    linearly interpolated.
    """
    if not query_step_embeddings:
        raise ValueError("query_step_embeddings must contain at least one step embedding.")

    step_losses: List[torch.Tensor] = []
    for step_emb in query_step_embeddings:
        step_loss, _ = compute_contrastive_loss(
            query_embeds=step_emb,
            pos_cand_embeds=pos_cand_embeds,
            neg_cand_embeds=neg_cand_embeds,
            temperature=temperature,
        )
        step_losses.append(step_loss)

    num_steps = len(step_losses)
    if num_steps == 1:
        # Degenerate case (K=0): keep single-step objective stable.
        step_weights = step_losses[0].new_tensor([1.0])
    else:
        step_weights = torch.linspace(
            float(min_step_weight),
            float(max_step_weight),
            steps=num_steps,
            device=step_losses[0].device,
            dtype=step_losses[0].dtype,
        )

    total_loss = torch.stack([
        w * l for w, l in zip(step_weights, step_losses)
    ]).sum()

    stats = {
        "loss_total": float(total_loss.detach().item()),
        "loss_step0": float(step_losses[0].detach().item()),
        "loss_stepk": float(step_losses[-1].detach().item()),
        "weight_step0": float(step_weights[0].detach().item()),
        "weight_stepk": float(step_weights[-1].detach().item()),
        "num_reason_steps": int(num_steps - 1),
        "step_losses": [float(x.detach().item()) for x in step_losses],
        "step_weights": [float(x.detach().item()) for x in step_weights],
    }
    return total_loss, stats

def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config, 
                    val_loader=None, eval_steps=0, model_without_ddp=None):
    """Train one epoch with per-sample truncation (no whole-batch dropping).
    
    Args:
        val_loader: Optional validation dataloader for step-level evaluation
        eval_steps: Number of steps between evaluations (0 to disable)
        model_without_ddp: Model without DDP wrapper for evaluation
    """
    # 1. 参数初始化
    max_token_length = getattr(config.model, 'max_token_length', 15000)
    max_visual_pixels = getattr(config.model, 'max_visual_pixels', 501760)
    save_steps = getattr(config.trainer_config, 'save_steps', 0)
    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    temperature = getattr(config.model, 'temperature', 0.07)
    step_weight_start = float(getattr(config.model, 'step_weight_start', 0.4))
    step_weight_end = float(getattr(config.model, 'step_weight_end', 1.0))
    debug_mode = _debug_enabled(config)
    debug_print_freq = int(getattr(config.trainer_config, 'debug_print_freq', 1))
    debug_max_text_chars = int(getattr(config.trainer_config, 'debug_max_text_chars', 1200))
    num_thought_tokens = getattr(config.model, 'num_thought_tokens', 0)
    enable_final_token = bool(getattr(config.model, 'enable_final_token', False))
    symmetric_qc_encoding = bool(getattr(config.model, 'symmetric_query_candidate_encoding', False))
    cand_num_thought_tokens = num_thought_tokens if symmetric_qc_encoding else 0
    cand_use_final_token = enable_final_token if symmetric_qc_encoding else False
    task_label = getattr(config.model, 'task_label', 'retrieval')
    print_freq = config.trainer_config.print_freq
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Train Epoch: [{}]".format(epoch)

    accumulation_counter = 0

    if utils.is_main_process():
        if symmetric_qc_encoding:
            print(
                "[Encoding Mode] Symmetric query/candidate encoding enabled: "
                f"TT={num_thought_tokens}, FT={enable_final_token} for both sides"
            )
        else:
            print(
                "[Encoding Mode] Asymmetric query/candidate encoding enabled: "
                f"query(TT={num_thought_tokens}, FT={enable_final_token}) vs candidate(TT=0, FT=False)"
            )
        if debug_mode:
            _debug_log(
                config,
                "Debug mode ON | "
                f"debug_print_freq={debug_print_freq}, "
                f"step_weight_start={step_weight_start}, step_weight_end={step_weight_end}, "
                f"accumulation_steps={accumulation_steps}",
            )

    import contextlib

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        unwrapped = model.module if hasattr(model, "module") else model
        index_mapping = batch["index_mapping"]
        query_indices = [idx[0] for idx in index_mapping["query"]]

        pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]
        neg_cand_indices = []
        if "neg_cand_list" in index_mapping:
            for neg_list in index_mapping["neg_cand_list"]:
                neg_cand_indices.extend(neg_list)

        # --- 确认通过检查，开始搬运数据 ---
        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
        image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)

        if debug_mode and utils.is_main_process() and (i % max(1, debug_print_freq) == 0):
            sample_query_text = ""
            if len(query_indices) > 0:
                try:
                    sample_query_text = str(txt_batched[query_indices[0]])
                except Exception:
                    sample_query_text = "<unavailable>"
            if len(sample_query_text) > debug_max_text_chars:
                sample_query_text = sample_query_text[:debug_max_text_chars] + "..."

            _debug_log(
                config,
                f"iter={i}, batch_size={image_batched.size(0)}, "
                f"query={len(query_indices)}, pos={len(pos_cand_indices)}, neg={len(neg_cand_indices)}, "
                f"txt_present={int(txt_mask.sum().item())}, img_present={int(image_mask.sum().item())}",
            )
            _debug_log(
                config,
                f"query_cfg: TT={num_thought_tokens}, FT={enable_final_token} | "
                f"cand_cfg: TT={cand_num_thought_tokens}, FT={cand_use_final_token}",
            )
            _debug_log(config, f"sample_query_text='{sample_query_text}'")

        # ================== [关键步骤 2：同步上下文与防御性梯度] ==================
        is_sync_step = (accumulation_counter + 1 == accumulation_steps)
        sync_context = model.no_sync() if not is_sync_step else contextlib.nullcontext()

        with sync_context:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # 正常的编码流程
                query_output = encode_batch_for_training(
                    model=model, txt_batched=txt_batched, image_batched=image_batched,
                    txt_mask=txt_mask, image_mask=image_mask, indices=query_indices,
                    task_label=task_label, device=gpu_id, num_thought_tokens=num_thought_tokens,
                    use_final_token=enable_final_token,
                    max_token_length=max_token_length, max_visual_pixels=max_visual_pixels,
                    return_reasoning_steps=True,
                )
                query_embeds = query_output.single_vec_emb

                pos_cand_embeds = encode_batch_for_training(
                    model=model, txt_batched=txt_batched, image_batched=image_batched,
                    txt_mask=txt_mask, image_mask=image_mask, indices=pos_cand_indices,
                    task_label=task_label, device=gpu_id, num_thought_tokens=cand_num_thought_tokens,
                    use_final_token=cand_use_final_token,
                    max_token_length=max_token_length, max_visual_pixels=max_visual_pixels
                )

                neg_cand_embeds = None
                if neg_cand_indices:
                    neg_cand_embeds = encode_batch_for_training(
                        model=model, txt_batched=txt_batched, image_batched=image_batched,
                        txt_mask=txt_mask, image_mask=image_mask, indices=neg_cand_indices,
                        task_label=task_label, device=gpu_id, num_thought_tokens=cand_num_thought_tokens,
                        use_final_token=cand_use_final_token,
                        max_token_length=max_token_length, max_visual_pixels=max_visual_pixels
                    )

                final_loss, inbatch_accuracy = compute_contrastive_loss(
                    query_embeds=query_embeds, pos_cand_embeds=pos_cand_embeds,
                    neg_cand_embeds=neg_cand_embeds, temperature=temperature
                )

                query_step_embeddings = query_output.thought_embeddings or [query_embeds]
                loss, reasoning_loss_stats = compute_multistep_reasoning_loss(
                    query_step_embeddings=query_step_embeddings,
                    pos_cand_embeds=pos_cand_embeds,
                    neg_cand_embeds=neg_cand_embeds,
                    temperature=temperature,
                    min_step_weight=step_weight_start,
                    max_step_weight=step_weight_end,
                )

                if debug_mode and utils.is_main_process() and (i % max(1, debug_print_freq) == 0):
                    step_count = 0 if query_output.thought_embeddings is None else len(query_output.thought_embeddings)
                    _debug_log(
                        config,
                        f"emb_shapes: query={tuple(query_embeds.shape)}, "
                        f"pos={tuple(pos_cand_embeds.shape)}, "
                        f"neg={None if neg_cand_embeds is None else tuple(neg_cand_embeds.shape)}, "
                        f"reason_steps={step_count}",
                    )
                    _debug_log(
                        config,
                        f"losses: total={float(loss.detach().item()):.6f}, "
                        f"step0={reasoning_loss_stats['loss_step0']:.6f}, "
                        f"stepk={reasoning_loss_stats['loss_stepk']:.6f}, "
                        f"w0={reasoning_loss_stats['weight_step0']:.3f}, "
                        f"wk={reasoning_loss_stats['weight_stepk']:.3f}, "
                        f"acc={float(inbatch_accuracy.detach().item()):.6f}",
                    )

            # 真实的 backward
            loss = loss / accumulation_steps
            # 只执行一次 backward，不需要 dummy_loss（DDP 的 find_unused_parameters=True 已处理未使用参数）
            loss.backward()

        # --- 优化器更新 ---
        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1
            
            # ✅ 新增：在更新前检查 special_embedding 的梯度状态，确保它真的在被训练
            if global_step % print_freq == 0 and utils.is_main_process():
                for name, param in model.named_parameters():
                    if "special_embedding" in name:
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            print(f"[Step {global_step}] {name} gradient norm: {grad_norm:.6f}")
                        else:
                            print(f"[Step {global_step}] {name} HAS NO GRADIENT! Check forward pass.")

            if debug_mode and utils.is_main_process() and (global_step % max(1, debug_print_freq) == 0):
                special_grad_norm = None
                lora_grad_norm = None
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if special_grad_norm is None and "special_embedding" in name:
                        special_grad_norm = float(param.grad.norm().item())
                    if lora_grad_norm is None and "lora" in name.lower():
                        lora_grad_norm = float(param.grad.norm().item())
                    if special_grad_norm is not None and lora_grad_norm is not None:
                        break
                _debug_log(
                    config,
                    f"step={global_step}, accumulation_synced={is_sync_step}, "
                    f"special_grad_norm={special_grad_norm}, lora_grad_norm={lora_grad_norm}",
                )

            # 标准梯度裁剪与更新
            # 注意：如果 param.grad 为 None，clip_grad_norm_ 也会正常忽略它
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulation_counter = 0

            # 记录日志
            if utils.is_main_process() and config.get("swanlab_config", {}).get("enabled", False):
                import swanlab
                swan_payload = {
                    "train/loss": loss.item() * accumulation_steps,
                    "train/loss_total": reasoning_loss_stats["loss_total"],
                    "train/loss_step0": reasoning_loss_stats["loss_step0"],
                    "train/loss_stepk": reasoning_loss_stats["loss_stepk"],
                    "train/weight_step0": reasoning_loss_stats["weight_step0"],
                    "train/weight_stepk": reasoning_loss_stats["weight_stepk"],
                    "train/num_reason_steps": reasoning_loss_stats["num_reason_steps"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/inbatch_accuracy": inbatch_accuracy.item(),
                    "train/grad_norm": total_norm.item(),
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }

                # 逐 step 记录 loss/weight，便于在 SwanLab 按 step 对齐观察。
                for idx, val in enumerate(reasoning_loss_stats.get("step_losses", [])):
                    swan_payload[f"train/loss_reason_step_{idx}"] = val
                for idx, val in enumerate(reasoning_loss_stats.get("step_weights", [])):
                    swan_payload[f"train/weight_reason_step_{idx}"] = val

                swanlab.log(swan_payload, step=global_step)

            # ================== [新增] Step-level 验证 ==================
            if eval_steps > 0 and global_step % eval_steps == 0:
                if val_loader is not None and model_without_ddp is not None:
                    if utils.is_main_process():
                        print(f"\n{'='*60}")
                        print(f"Running validation at step {global_step}...")
                        print(f"{'='*60}")
                    
                    val_stats = eval_engine(model_without_ddp, model, val_loader, gpu_id, config)
                    
                    if utils.is_main_process():
                        print(f"Validation Results (Step {global_step}):")
                        for key, value in val_stats.items():
                            print(f"  {key}: {value:.6f}")
                        print(f"{'='*60}\n")
                        
                        # Log validation metrics to SwanLab if enabled
                        if config.get("swanlab_config", {}).get("enabled", False):
                            import swanlab
                            val_metrics = {f"val/{key}": value for key, value in val_stats.items()}
                            swanlab.log(val_metrics, step=global_step)
            # ==============================================================            # 保存模型
            if save_steps > 0 and global_step % save_steps == 0:
                if utils.is_main_process():
                    from models.qwen3a.train import save_step_checkpoint
                    save_step_checkpoint(model, optimizer, scheduler, epoch, global_step, scaler, config)

        metric_logger.update(loss=loss.item() * accumulation_steps)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # Removed redundant barrier - synchronize_between_processes already synchronizes via all_reduce
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step



@torch.no_grad()
def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
    """
    Evaluate Qwen3-VL model using contrastive loss metrics.
    """
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    # Task label for compatibility
    task_label = getattr(config.model, 'task_label', 'retrieval')
    temperature = getattr(config.model, 'temperature', 0.07)
    step_weight_start = float(getattr(config.model, 'step_weight_start', 0.4))
    step_weight_end = float(getattr(config.model, 'step_weight_end', 1.0))
    debug_mode = _debug_enabled(config)
    debug_print_freq = int(getattr(config.trainer_config, 'debug_print_freq', 1)) if hasattr(config, 'trainer_config') else 1
    debug_max_text_chars = int(getattr(config.trainer_config, 'debug_max_text_chars', 1200)) if hasattr(config, 'trainer_config') else 120
    task_label = getattr(config.model, 'mbeir_task_label', task_label)
    num_thought_tokens = getattr(config.model, 'num_thought_tokens', 0)
    enable_final_token = bool(getattr(config.model, 'enable_final_token', False))
    symmetric_qc_encoding = bool(getattr(config.model, 'symmetric_query_candidate_encoding', False))
    cand_num_thought_tokens = num_thought_tokens if symmetric_qc_encoding else 0
    cand_use_final_token = enable_final_token if symmetric_qc_encoding else False
    reason_steps = getattr(config.model, 'reason_steps', 0)
    use_cache_for_reasoning = getattr(config.model, 'use_cache_for_eval', True)
    max_token_length = getattr(config.model, 'max_token_length', 15000)
    max_visual_pixels = getattr(config.model, 'max_visual_pixels', 501760)

    if num_thought_tokens > 0:
        print(f"Evaluating with num_thought_tokens={num_thought_tokens}")
    if reason_steps > 0:
        print(f"Evaluating with reason_steps={reason_steps} (Note: Qwen3-VL doesn't use reasoning steps)")
    if symmetric_qc_encoding:
        print(
            "Evaluating with symmetric query/candidate encoding: "
            f"TT={num_thought_tokens}, FT={enable_final_token} for both sides"
        )
    else:
        print(
            "Evaluating with asymmetric query/candidate encoding: "
            f"query(TT={num_thought_tokens}, FT={enable_final_token}) vs candidate(TT=0, FT=False)"
        )
    if debug_mode and utils.is_main_process():
        _debug_log(
            config,
            "Eval debug ON | "
            f"debug_print_freq={debug_print_freq}, "
            f"step_weight_start={step_weight_start}, step_weight_end={step_weight_end}, "
            f"task_label={task_label}",
        )

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move tensors to GPU
        txt_batched = batch["txt_batched"]

        # Keep image_batched on CPU
        image_batched = batch["image_batched"]

        txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
        image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)
        index_mapping = batch["index_mapping"]

        # Flatten indices
        query_indices = [idx[0] for idx in index_mapping["query"]]
        pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]

        if debug_mode and utils.is_main_process() and (i % max(1, debug_print_freq) == 0):
            sample_query_text = ""
            if len(query_indices) > 0:
                try:
                    sample_query_text = str(txt_batched[query_indices[0]])
                except Exception:
                    sample_query_text = "<unavailable>"
            if len(sample_query_text) > debug_max_text_chars:
                sample_query_text = sample_query_text[:debug_max_text_chars] + "..."
            _debug_log(
                config,
                f"eval_iter={i}, batch_size={image_batched.size(0)}, "
                f"query={len(query_indices)}, pos={len(pos_cand_indices)}, "
                f"txt_present={int(txt_mask.sum().item())}, img_present={int(image_mask.sum().item())}",
            )
            _debug_log(
                config,
                f"eval_cfg: query(TT={num_thought_tokens}, FT={enable_final_token}) | "
                f"cand(TT={cand_num_thought_tokens}, FT={cand_use_final_token})",
            )
            _debug_log(config, f"eval_sample_query_text='{sample_query_text}'")

        # autocast for mixed precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Encode queries with thought tokens
            query_output = encode_batch_for_training(
                model=model,
                txt_batched=txt_batched,
                image_batched=image_batched,
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=query_indices,
                task_label=task_label,
                device=gpu_id,
                num_thought_tokens=num_thought_tokens,
                use_final_token=enable_final_token,
                reason_steps=reason_steps,
                use_cache_for_reasoning=use_cache_for_reasoning,
                max_token_length=max_token_length,
                max_visual_pixels=max_visual_pixels,
                return_reasoning_steps=True,
            )
            query_embeds = query_output.single_vec_emb

            # Encode positive candidates without thought tokens
            pos_cand_embeds = encode_batch_for_training(
                model=model,
                txt_batched=txt_batched,
                image_batched=image_batched,
                txt_mask=txt_mask,
                image_mask=image_mask,
                indices=pos_cand_indices,
                task_label=task_label,
                device=gpu_id,
                num_thought_tokens=cand_num_thought_tokens,
                use_final_token=cand_use_final_token,
                reason_steps=0,
                use_cache_for_reasoning=use_cache_for_reasoning,
                max_token_length=max_token_length,
                max_visual_pixels=max_visual_pixels,
            )

            # Compute final-step contrastive loss
            final_loss, inbatch_accuracy = compute_contrastive_loss(
                query_embeds=query_embeds,
                pos_cand_embeds=pos_cand_embeds,
                neg_cand_embeds=None,
                temperature=temperature,
            )

            # Keep evaluation objective compatible with training objective.
            query_step_embeddings = query_output.thought_embeddings or [query_embeds]
            loss, reasoning_loss_stats = compute_multistep_reasoning_loss(
                query_step_embeddings=query_step_embeddings,
                pos_cand_embeds=pos_cand_embeds,
                neg_cand_embeds=None,
                temperature=temperature,
                min_step_weight=step_weight_start,
                max_step_weight=step_weight_end,
            )

            if debug_mode and utils.is_main_process() and (i % max(1, debug_print_freq) == 0):
                step_count = 0 if query_output.thought_embeddings is None else len(query_output.thought_embeddings)
                _debug_log(
                    config,
                    f"eval_emb_shapes: query={tuple(query_embeds.shape)}, "
                    f"pos={tuple(pos_cand_embeds.shape)}, reason_steps={step_count}",
                )
                _debug_log(
                    config,
                    f"eval_losses: total={float(loss.detach().item()):.6f}, "
                    f"step0={reasoning_loss_stats['loss_step0']:.6f}, "
                    f"stepk={reasoning_loss_stats['loss_stepk']:.6f}, "
                    f"w0={reasoning_loss_stats['weight_step0']:.3f}, "
                    f"wk={reasoning_loss_stats['weight_stepk']:.3f}, "
                    f"acc={float(inbatch_accuracy.detach().item()):.6f}",
                )

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # Ensure all ranks complete evaluation loop before synchronizing
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
