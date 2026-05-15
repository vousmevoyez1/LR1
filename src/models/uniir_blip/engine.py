import torch
from torch.cuda.amp import autocast
import transformers


from models.uniir_blip import utils


def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    model.train()

    ### 构建度量记录器，定义要跟踪的指标：学习率、loss、in-batch accuracy；window_size=1 表示不做滑窗平滑。
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0
    ### 迭代 dataloader，并由 log_every 控制打印频率。
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key, value in batch.items():
            #### 换成jina v4后需要考虑这里的修改
            ### 将 batch 中的张量或 HF BatchEncoding 内部张量搬到当前 GPU。
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(gpu_id, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(gpu_id)

        ### alpha 线性 warmup（仅第 0 个 epoch）：从 0 线性涨到设定 alpha，之后各 epoch 固定为 alpha。
        ### alpha 控制队列对比损失的权重。
        if epoch > 0:
            alpha = config.model.alpha
        else:
            alpha = config.model.alpha * min(1, i / len(data_loader))

        # autocast for mixed precision
        with autocast():
            outputs = model(batch=batch, alpha=alpha)
            loss = outputs["loss"]
            inbatch_accuracy = outputs["accuracy"]

        # Scale the loss by the number of accumulation steps since backward averages the gradients.
        ### 在 backward 前缩放 loss：累积 accumulation_steps 次后，整体有效梯度与不缩放一次 step 等价。
        loss = loss / accumulation_steps

        # Use scaler for backward
        ### AMP 的缩放反传：避免半精下梯度下溢。
        scaler.scale(loss).backward()

        ### 当累积次数达到阈值：
        # 	•	步进计数（global_step 这里仅内部累加，外部未使用）。
        # 	•	scaler.step(optimizer)：尝试更新参数；若出现 inf/nan，step 会跳过。
        # 	•	scaler.update()：调整缩放系数。
        # 	•	model.zero_grad()：清空梯度（✅ 建议改为 optimizer.zero_grad(set_to_none=True)，更省内存、快）。
        # 	•	scheduler.step()：每个优化步后调度学习率。
        #       请确保 CosineAnnealingLR(T_max) 的 T_max 按优化步计算（主程序已按迭代步计算，合理）。
        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            model.zero_grad()
            scheduler.step()
            accumulation_counter = 0

        metric_logger.update(loss=loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # TODO: might need to loop through all param groups
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    ### 多进程聚合指标（例如取均值），打印并返回每个 meter 的全局平均。
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

### 评估（不反传）
@torch.no_grad()
def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    # Save model states
    saved_state = model_without_ddp.state_dict()

    # Clear model queue states
    ### 关键：清理对比学习队列（query/cand/idx 队列与指针），避免训练阶段的队列状态污染评估。
    model_without_ddp.query_queue.copy_(torch.randn_like(model_without_ddp.query_queue))
    model_without_ddp.cand_queue.copy_(torch.randn_like(model_without_ddp.cand_queue))
    model_without_ddp.idx_queue.copy_(torch.full_like(model_without_ddp.idx_queue, -100))
    model_without_ddp.new_ptr_queue.zero_()
    print("Cleared model queue states.")

    ### 与训练同样，把 batch 搬到 GPU。
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(gpu_id, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(gpu_id)

        alpha = config.model.alpha * min(1, i / len(data_loader))

        # autocast for mixed precision
        with autocast():
            outputs = model(batch=batch, alpha=alpha)
            loss = outputs["loss"]
            inbatch_accuracy = outputs["accuracy"]

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    # Restore model states from the saved variables
    model_without_ddp.load_state_dict(saved_state)
    print("Restored model queue states and model states from the saved variables.")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
