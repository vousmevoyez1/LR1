import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional, Union

from models.qwen3 import utils


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
    from models.qwen3.qwen3_thought_wrapper import Qwen3ThoughtOutput

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

    # Process image-only samples
    # 2. 修改 Text-only 处理段
    # if text_only_idx:
    #     local_indices = [x[0] for x in text_only_idx]
    #     global_indices = [x[1] for x in text_only_idx]
    #     # texts = txt_batched[global_indices]
    #     # [新增] 粗糙的 CPU 级文本截断 (按字符数截断，防止超长文本撑爆 Tokenizer)
    #     # 假设 1 个 Token 大约等于 2-3 个字符，保守截断
    #     char_limit = 5000
    #     texts = [txt_batched[idx][:char_limit] for idx in global_indices]

    #     # 预处理：调用 unwrapped_model
    #     inputs = unwrapped_model._preprocess_inputs(texts=texts, images=None, num_thought_tokens=num_thought_tokens,max_token_length=max_token_length,       # [传入]
    #         max_visual_pixels=max_visual_pixels )     # [传入])
    #     inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    #     # 前向传播：必须直接调用带 DDP 壳子的 model(xxx)，触发 forward！
    #     output = model(encode_mbeir_batch=False, num_thought_tokens=num_thought_tokens, **inputs)
    #     embeddings[local_indices] = output.single_vec_emb.to(model_dtype)

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
        # inputs = unwrapped_model._preprocess_inputs(
        #     texts=[""] * len(pil_images),
        #     images=pil_images,
        #     num_thought_tokens=num_thought_tokens,
        #     max_token_length=max_token_length,       # [传入]
        #     max_visual_pixels=max_visual_pixels      # [传入]
        # )
        # inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        # # 触发 DDP 的 forward
        # output = model(encode_mbeir_batch=False, num_thought_tokens=num_thought_tokens, **inputs)
        # embeddings[local_indices] = output.single_vec_emb.to(model_dtype)

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
        # inputs = unwrapped_model._preprocess_inputs(
        #     texts=texts,
        #     images=pil_images,
        #     num_thought_tokens=num_thought_tokens,
        #     max_token_length=max_token_length,       # [传入]
        #     max_visual_pixels=max_visual_pixels      # [传入]
        # )
        # inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        # # 触发 DDP 的 forward
        # output = model(encode_mbeir_batch=False, num_thought_tokens=num_thought_tokens, **inputs)
        # embeddings[local_indices] = output.single_vec_emb.to(model_dtype)

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
    final_loss: torch.Tensor,
    thought_step_embeddings: Optional[List[torch.Tensor]],
    pos_cand_embeds: torch.Tensor,
    neg_cand_embeds: Optional[torch.Tensor],
    temperature: float,
    lambda_ds: float,
    gamma_ds: float,
    lambda_prog: float,
    progressive_margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-step reasoning loss:
    L_total = L_F + lambda_ds * sum_{i=1}^{K-1} gamma^{K-i} * L_i
              + lambda_prog * L_prog
    """
    stats = {
        "loss_final": float(final_loss.detach().item()),
        "loss_ds": 0.0,
        "loss_prog": 0.0,
    }

    if not thought_step_embeddings:
        return final_loss, stats

    step_losses: List[torch.Tensor] = []
    for step_emb in thought_step_embeddings:
        step_loss, _ = compute_contrastive_loss(
            query_embeds=step_emb,
            pos_cand_embeds=pos_cand_embeds,
            neg_cand_embeds=neg_cand_embeds,
            temperature=temperature,
        )
        step_losses.append(step_loss)

    if len(step_losses) == 0:
        return final_loss, stats

    if len(step_losses) >= 2:
        # Use all but the last step (the last is final token) for deep supervision.
        k = len(step_losses)
        ds_terms = []
        for i in range(k - 1):
            # i is 0-based -> formula exponent uses step index in [1..K-1]
            exponent = k - (i + 1)
            weight = gamma_ds ** exponent
            ds_terms.append(weight * step_losses[i])
        loss_ds = torch.stack(ds_terms).sum() if ds_terms else final_loss.new_zeros(())
    else:
        loss_ds = final_loss.new_zeros(())

    loss_prog = compute_reasoning_progressive_regularizer(step_losses, progressive_margin)

    total_loss = final_loss + lambda_ds * loss_ds + lambda_prog * loss_prog

    stats["loss_ds"] = float(loss_ds.detach().item())
    stats["loss_prog"] = float(loss_prog.detach().item())
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
    lambda_ds = float(getattr(config.model, 'lambda_ds', 1.0))
    gamma_ds = float(getattr(config.model, 'gamma_ds', 0.8))
    lambda_prog = float(getattr(config.model, 'lambda_prog', 0.0))
    progressive_margin = float(getattr(config.model, 'progressive_margin', 0.0))
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
                f"lambda_ds={lambda_ds}, gamma_ds={gamma_ds}, "
                f"lambda_prog={lambda_prog}, progressive_margin={progressive_margin}, "
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

                loss, reasoning_loss_stats = compute_multistep_reasoning_loss(
                    final_loss=final_loss,
                    thought_step_embeddings=query_output.thought_embeddings,
                    pos_cand_embeds=pos_cand_embeds,
                    neg_cand_embeds=neg_cand_embeds,
                    temperature=temperature,
                    lambda_ds=lambda_ds,
                    gamma_ds=gamma_ds,
                    lambda_prog=lambda_prog,
                    progressive_margin=progressive_margin,
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
                        f"final={reasoning_loss_stats['loss_final']:.6f}, "
                        f"ds={reasoning_loss_stats['loss_ds']:.6f}, "
                        f"prog={reasoning_loss_stats['loss_prog']:.6f}, "
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
                swanlab.log({
                    "train/loss": loss.item() * accumulation_steps,
                    "train/loss_final": reasoning_loss_stats["loss_final"],
                    "train/loss_ds": reasoning_loss_stats["loss_ds"],
                    "train/loss_prog": reasoning_loss_stats["loss_prog"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/inbatch_accuracy": inbatch_accuracy.item(),
                    "train/grad_norm": total_norm.item()
                }, step=global_step)

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
                    from models.qwen3.train import save_step_checkpoint
                    save_step_checkpoint(model, optimizer, scheduler, epoch, global_step, scaler, config)

        metric_logger.update(loss=loss.item() * accumulation_steps)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # Removed redundant barrier - synchronize_between_processes already synchronizes via all_reduce
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step

# def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
#     """
#     Train Qwen3-VL model for one epoch using contrastive loss.

#     The model is a Qwen3VLEmbedder instance.
#     Uses single-vector embeddings for contrastive learning.

#     Returns:
#         Tuple of (train_stats dict, updated global_step)
#     """
#     # 找到获取 config 的地方，确保有这两个参数
#     max_token_length = getattr(config.model, 'max_token_length', 2048)
#     # 建议 5090 (32GB) 设为 501760；如果还是 OOM，降到 250880
#     max_visual_pixels = getattr(config.model, 'max_visual_pixels', 501760)
#     model.train()

#     # Metric logger setup
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     header = "Train Epoch: [{}]".format(epoch)
#     print_freq = config.trainer_config.print_freq

#     accumulation_steps = config.trainer_config.gradient_accumulation_steps
#     accumulation_counter = 0

#     # Task label for compatibility (not used in Qwen3-VL)
#     task_label = getattr(config.model, 'task_label', 'retrieval')
#     temperature = getattr(config.model, 'temperature', 0.07)
#     num_thought_tokens = getattr(config.model, 'num_thought_tokens', 0)
#     reason_steps = getattr(config.model, 'reason_steps', 0)
#     max_token_length = getattr(config.model, 'max_token_length', None)
#     save_steps = getattr(config.trainer_config, 'save_steps', 0)

#     if num_thought_tokens > 0:
#         print(f"Training with num_thought_tokens={num_thought_tokens} (query uses thought tokens, candidates don't)")
#     if reason_steps > 0:
#         print(f"Training with reason_steps={reason_steps} (Note: Qwen3-VL doesn't use reasoning steps)")
#     if max_token_length is not None:
#         print(f"Using max_token_length={max_token_length} for input truncation")

#     import contextlib

#     for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         unwrapped = model.module if hasattr(model, "module") else model
        
#         # 找到视觉塔的一个参数（比如第一层权重）
#         visual_param = next(unwrapped.model.visual.parameters())
#         dummy_loss = (visual_param.sum() * 0.0)
        
#         # 加上 LoRA 和新增 Token 的参数，做一个全员大礼包
#         dummy_loss += sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
        
#         # 立即执行一次 backward，给 NCCL 派发所有层的“通关文牒”
#         dummy_loss.backward()
#         # Move tensors to GPU
#         txt_batched = batch["txt_batched"]  # RawTextBatch

#         # Keep image_batched on CPU for PIL conversion
#         image_batched = batch["image_batched"]

#         # Move masks to GPU as they are needed for indexing
#         txt_mask = batch["txt_mask_batched"].to(gpu_id, non_blocking=True)
#         image_mask = batch["image_mask_batched"].to(gpu_id, non_blocking=True)
#         index_mapping = batch["index_mapping"]

#         batch_size = len(index_mapping["query"])

#         # Flatten indices for query, pos_cand, neg_cand
#         query_indices = [idx[0] for idx in index_mapping["query"]]
#         pos_cand_indices = [idx[0] for idx in index_mapping["pos_cand"]]

#         # Handle negative candidates if present
#         neg_cand_indices = []
#         if "neg_cand_list" in index_mapping:
#             for neg_list in index_mapping["neg_cand_list"]:
#                 neg_cand_indices.extend(neg_list)

#         # ================== [防死锁核心逻辑] ==================
#         # 判断是否是梯度累积的最后一步
#         is_sync_step = (accumulation_counter + 1 == accumulation_steps)
#         # 如果不是最后一步，启用 no_sync() 挂起卡间通信
#         sync_context = model.no_sync() if not is_sync_step else contextlib.nullcontext()

#         # 将前向传播和反向传播包裹在 sync_context 中
#         with sync_context:
#             # autocast for mixed precision (修复了 API 警告)
#             with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 # Encode queries with thought tokens
#                 query_embeds = encode_batch_for_training(
#                     model=model,
#                     txt_batched=txt_batched,
#                     image_batched=image_batched,
#                     txt_mask=txt_mask,
#                     image_mask=image_mask,
#                     indices=query_indices,
#                     task_label=task_label,
#                     device=gpu_id,
#                     num_thought_tokens=num_thought_tokens,
#                     reason_steps=reason_steps,
#                     use_cache_for_reasoning=False,
#                     max_token_length=max_token_length,
#                 )

#                 # Encode positive candidates without thought tokens
#                 pos_cand_embeds = encode_batch_for_training(
#                     model=model,
#                     txt_batched=txt_batched,
#                     image_batched=image_batched,
#                     txt_mask=txt_mask,
#                     image_mask=image_mask,
#                     indices=pos_cand_indices,
#                     task_label=task_label,
#                     device=gpu_id,
#                     num_thought_tokens=0,
#                     reason_steps=0,
#                     use_cache_for_reasoning=False,
#                     max_token_length=max_token_length,
#                 )

#                 # Encode negative candidates if present (without thought tokens)
#                 neg_cand_embeds = None
#                 if neg_cand_indices:
#                     neg_cand_embeds = encode_batch_for_training(
#                         model=model,
#                         txt_batched=txt_batched,
#                         image_batched=image_batched,
#                         txt_mask=txt_mask,
#                         image_mask=image_mask,
#                         indices=neg_cand_indices,
#                         task_label=task_label,
#                         device=gpu_id,
#                         num_thought_tokens=0,
#                         reason_steps=0,
#                         use_cache_for_reasoning=False,
#                         max_token_length=max_token_length,
#                     )

#                 # Compute contrastive loss
#                 loss, inbatch_accuracy = compute_contrastive_loss(
#                     query_embeds=query_embeds,
#                     pos_cand_embeds=pos_cand_embeds,
#                     neg_cand_embeds=neg_cand_embeds,
#                     temperature=temperature,
#                 )

#             # Scale the loss for gradient accumulation
#             loss = loss / accumulation_steps

#             # Backward pass
#             #scaler.scale(loss).backward()
#             loss.backward()
#             # 【注意】：去掉了你原来在这里写的 clip_grad_norm_
#             # 梯度裁剪必须在 scaler.unscale_() 之后执行，否则会导致混合精度崩溃！

#         # ==============================================================

#         accumulation_counter += 1
#         if accumulation_counter == accumulation_steps:
#             global_step += 1

#             # [修改] 标准的 scaler 梯度裁剪与更新流程
#             #scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             # scaler.step(optimizer)
#             # scaler.update()
#             total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             print(f"Step: {global_step}, Grad Norm: {total_norm}")
#             optimizer.step()
#             optimizer.zero_grad(set_to_none=True)
#             scheduler.step()
#             accumulation_counter = 0

#             # ================== [新增] 实时回传 Loss 到大屏 ==================
#             if utils.is_main_process() and config.get("swanlab_config", {}).get("enabled", False):
#                 import swanlab
#                 swanlab.log({
#                     "train/loss": loss.item() * accumulation_steps,  # 还原真实的 batch loss
#                     "train/lr": optimizer.param_groups[0]["lr"],
#                     "train/inbatch_accuracy": inbatch_accuracy.item()
#                 }, step=global_step)
#             # ==============================================================

#             # Step-level checkpoint saving
#             if save_steps > 0 and global_step % save_steps == 0:
#                 if utils.is_main_process():
#                     from models.qwen3.train import save_step_checkpoint
#                     save_step_checkpoint(
#                         model, optimizer, scheduler,
#                         epoch, global_step, scaler, config
#                     )

#         metric_logger.update(loss=loss.item() * accumulation_steps)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

#     # Ensure all ranks complete training loop before synchronizing
#     if torch.distributed.is_initialized():
#         torch.distributed.barrier()

#     # Gather stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger.global_avg())

#     # Return both stats and global_step for tracking
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step


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
    lambda_ds = float(getattr(config.model, 'lambda_ds', 1.0))
    gamma_ds = float(getattr(config.model, 'gamma_ds', 0.8))
    lambda_prog = float(getattr(config.model, 'lambda_prog', 0.0))
    progressive_margin = float(getattr(config.model, 'progressive_margin', 0.0))
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
            f"lambda_ds={lambda_ds}, gamma_ds={gamma_ds}, "
            f"lambda_prog={lambda_prog}, progressive_margin={progressive_margin}, "
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
            loss, reasoning_loss_stats = compute_multistep_reasoning_loss(
                final_loss=final_loss,
                thought_step_embeddings=query_output.thought_embeddings,
                pos_cand_embeds=pos_cand_embeds,
                neg_cand_embeds=None,
                temperature=temperature,
                lambda_ds=lambda_ds,
                gamma_ds=gamma_ds,
                lambda_prog=lambda_prog,
                progressive_margin=progressive_margin,
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
                    f"final={reasoning_loss_stats['loss_final']:.6f}, "
                    f"ds={reasoning_loss_stats['loss_ds']:.6f}, "
                    f"prog={reasoning_loss_stats['loss_prog']:.6f}, "
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
