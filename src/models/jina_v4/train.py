# Standard library
import argparse
import logging
import os
import random
import time
import datetime
import signal
import sys
import atexit

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
### 混合精度（AMP）下的梯度缩放，避免下溢
from torch.cuda.amp import GradScaler
### 余弦退火学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR
### 加载/操作 YAML 风格配置
from omegaconf import OmegaConf
from dotenv import load_dotenv
import swanlab

# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_distributed_sampler_list,
    build_dataloader_list,
)

from models.jina_v4.engine import train_one_epoch, eval_engine
import models.jina_v4.utils as utils

from models.jina_v4.jina_v4.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model


# Set up logger
logger = logging.getLogger()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_trainable_parameters(model, model_name="Model"):
    """Print parameter statistics for the model."""
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"{model_name} Parameter Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {total_params - trainable_params:,}")
    print(f"  Trainable ratio:      {100 * trainable_params / total_params:.4f}%")
    print(f"{'='*60}\n")
    return total_params, trainable_params


def freeze_base_model_keep_lora(model):
    """
    Freeze all parameters except LoRA layers.
    
    For PeftModel (LoRA-wrapped model), this freezes the base model
    and keeps only the LoRA adapter parameters trainable.
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    # LoRA parameters are typically named with 'lora_A', 'lora_B', or 'lora_'
    lora_param_count = 0
    for name, param in model.named_parameters():
        if 'lora_' in name.lower():
            param.requires_grad = True
            lora_param_count += param.numel()
    
    print(f"Froze base model. LoRA parameters unfrozen: {lora_param_count:,}")
    return model


# Global variables for signal handling
_checkpoint_state = {
    "model": None,
    "optimizer": None,
    "scheduler": None,
    "epoch": 0,
    "global_step": 0,
    "scaler": None,
    "config": None,
    "should_stop": False,
}


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT signals for graceful shutdown."""
    signal_name = signal.Signals(signum).name
    print(f"\n{'='*60}")
    print(f"Received {signal_name} signal. Saving checkpoint before exit...")
    print(f"{'='*60}")
    
    _checkpoint_state["should_stop"] = True
    
    # Only save on main process
    if utils.is_main_process() and _checkpoint_state["model"] is not None:
        save_emergency_checkpoint(
            _checkpoint_state["model"],
            _checkpoint_state["optimizer"],
            _checkpoint_state["scheduler"],
            _checkpoint_state["epoch"],
            _checkpoint_state["global_step"],
            _checkpoint_state["scaler"],
            _checkpoint_state["config"],
        )
    
    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()
    
    sys.exit(0)


def save_emergency_checkpoint(model, optimizer, scheduler, epoch, global_step, scaler, config):
    """Save emergency checkpoint on signal interrupt."""
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_emergency_epoch{epoch}_step{global_step}.pth"
    
    model_to_save = model.module if hasattr(model, 'module') else model
    
    if hasattr(model_to_save, 'get_adapter_state_dict'):
        adapter_state_dict = {}
        for name, param in model_to_save.named_parameters():
            if param.requires_grad:
                adapter_state_dict[name] = param.data.cpu()
        
        save_obj = {
            "lora_adapter": adapter_state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
            "scaler": scaler.state_dict(),
            "is_emergency": True,
        }
    else:
        save_obj = {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
            "scaler": scaler.state_dict(),
            "is_emergency": True,
        }
    
    checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"[EMERGENCY] Saved checkpoint to {checkpoint_path}")


def save_step_checkpoint(model, optimizer, scheduler, epoch, global_step, scaler, config):
    """
    Save step-level checkpoint for resuming mid-epoch training.
    Also manages checkpoint rotation to limit disk usage.
    """
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_step_{global_step}.pth"
    
    model_to_save = model.module if hasattr(model, 'module') else model
    
    if hasattr(model_to_save, 'get_adapter_state_dict'):
        adapter_state_dict = {}
        for name, param in model_to_save.named_parameters():
            if param.requires_grad:
                adapter_state_dict[name] = param.data.cpu()
        
        save_obj = {
            "lora_adapter": adapter_state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
            "scaler": scaler.state_dict(),
        }
    else:
        save_obj = {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
            "scaler": scaler.state_dict(),
        }
    
    ckpt_dir = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, checkpoint_name)
    torch.save(save_obj, checkpoint_path)
    print(f"[Step {global_step}] Saved checkpoint to {checkpoint_path}")
    
    # Rotate old step checkpoints if save_total_limit is set
    save_total_limit = getattr(config.trainer_config, 'save_total_limit', 0)
    if save_total_limit > 0:
        _rotate_step_checkpoints(ckpt_dir, model_name, save_total_limit)


def _rotate_step_checkpoints(ckpt_dir, model_name, save_total_limit):
    """Remove old step checkpoints, keeping only the most recent ones."""
    import glob
    pattern = os.path.join(ckpt_dir, f"{model_name}_step_*.pth")
    step_ckpts = glob.glob(pattern)
    
    if len(step_ckpts) > save_total_limit:
        # Sort by step number (extract from filename)
        def get_step(path):
            basename = os.path.basename(path)
            try:
                return int(basename.split('_step_')[1].replace('.pth', ''))
            except:
                return 0
        
        step_ckpts.sort(key=get_step)
        
        # Remove oldest checkpoints
        for old_ckpt in step_ckpts[:-save_total_limit]:
            try:
                os.remove(old_ckpt)
                print(f"[Rotation] Removed old checkpoint: {old_ckpt}")
            except Exception as e:
                print(f"[Rotation] Failed to remove {old_ckpt}: {e}")


def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config):
    """
    Save checkpoint with LoRA adapter weights.
    
    For Jina V4 (PeftModel), we save only the LoRA adapter state dict
    along with optimizer and scheduler states.
    """
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    
    # Get the model without DDP wrapper if needed
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # For PEFT model, save adapter weights separately
    # This is more memory efficient and allows easier loading
    if hasattr(model_to_save, 'get_adapter_state_dict'):
        # PEFT model - save only adapter weights
        adapter_state_dict = {}
        for name, param in model_to_save.named_parameters():
            if param.requires_grad:  # Only save trainable (LoRA) parameters
                adapter_state_dict[name] = param.data.cpu()
        
        save_obj = {
            "lora_adapter": adapter_state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "scaler": scaler.state_dict(),
        }
    else:
        # Fallback: save full model state
        save_obj = {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "scaler": scaler.state_dict(),
        }
    
    checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_lora_checkpoint(model, checkpoint_path):
    """
    Load LoRA adapter weights from a checkpoint.
    
    Handles both new-style checkpoints (with 'lora_adapter' key) and
    old-style checkpoints (with full 'model' state dict).
    
    Args:
        model: The PeftModel (LoRA-wrapped model)
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model: Model with loaded LoRA weights
        checkpoint: Full checkpoint dict (for optimizer/scheduler loading)
    """
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"Checkpoint file {checkpoint_path} does not exist")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "lora_adapter" in checkpoint:
        # New-style checkpoint with only LoRA weights
        adapter_state_dict = checkpoint["lora_adapter"]
        
        # Load LoRA weights
        model_state_dict = model.state_dict()
        for name, param in adapter_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
            else:
                print(f"Warning: LoRA parameter {name} not found in model")
        
        print(f"Loaded LoRA adapter weights from {checkpoint_path}")
        
    elif "model" in checkpoint:
        # Old-style checkpoint with full model state
        state_dict = checkpoint["model"]
        
        # Only load LoRA-related parameters
        model_state_dict = model.state_dict()
        lora_keys_loaded = 0
        for name, param in state_dict.items():
            if 'lora_' in name.lower() and name in model_state_dict:
                model_state_dict[name].copy_(param)
                lora_keys_loaded += 1
        
        print(f"Loaded {lora_keys_loaded} LoRA parameters from {checkpoint_path}")
    else:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no 'lora_adapter' or 'model' key")
    
    return model, checkpoint


def log_results(train_stats, val_stats, test_stats, epoch=None, best_epoch=None):
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats


def train(
    train_loader,
    val_loader,
    model,
    model_without_ddp,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch,
):
    gpu_id = config.dist_config.gpu_id
    is_distributed_mode = config.dist_config.distributed_mode
    
    # Register signal handlers for graceful shutdown (only on main process)
    if utils.is_main_process():
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        print("Registered signal handlers for graceful checkpoint saving")
    
    # Initialize checkpoint state for signal handler
    _checkpoint_state["model"] = model_without_ddp
    _checkpoint_state["optimizer"] = optimizer
    _checkpoint_state["scheduler"] = scheduler
    _checkpoint_state["scaler"] = scaler
    _checkpoint_state["config"] = config
    
    ### best_inbatch_accuracy 用于追踪最优 in-batch acc。
    global_step, total_loss, best_inbatch_accuracy = (
        0,
        0.0,
        0.0,
    )
    best_epoch = 0
    model.zero_grad()
    
    # Get save_steps config (0 means disabled)
    save_steps = getattr(config.trainer_config, 'save_steps', 0)
    if save_steps > 0:
        print(f"Step-level checkpointing enabled: saving every {save_steps} optimizer steps")

    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        # Update checkpoint state
        _checkpoint_state["epoch"] = epoch
        
        # Set different seed for different epoch
        ### DDP 关键点：每个 epoch 为 DistributedSampler 设置新种子，确保各 rank 的 shuffle 一致。
        if is_distributed_mode:
            train_loader.sampler.set_epoch(epoch)

        # Modified: train_one_epoch now returns (train_stats, global_step)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            gpu_id,
            scheduler,
            global_step,
            scaler,
            config,
        )
        
        # Handle both old (dict only) and new (dict, int) return formats
        if isinstance(train_result, tuple):
            train_stats, global_step = train_result
        else:
            train_stats = train_result
            # Estimate global_step if not returned
            steps_per_epoch = len(train_loader) // config.trainer_config.gradient_accumulation_steps
            global_step = (epoch + 1) * steps_per_epoch
        
        # Update checkpoint state
        _checkpoint_state["global_step"] = global_step

        eval_freq = config.evaluator.eval_freq
        ### 若无 val_loader（评估关闭）或未到评估频次，仅记录 train 结果并仍然保存 checkpoint（保障每个 epoch 的存档）。
        if val_loader is None or epoch % eval_freq != 0:
            log_stats = log_results(train_stats, None, None, epoch, best_epoch)
            ### 只在主进程保存，避免多进程竞争文件。
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
        else:
            ### eval_engine 负责评估，返回字典，代码期望有 "inbatch_accuracy"
            val_status = eval_engine(model_without_ddp, model, val_loader, gpu_id, config)
            try:
                inbatch_accuracy = float(val_status["inbatch_accuracy"])
            except ValueError:
                print(f"Error: Expected a number but got '{val_status['inbatch_accuracy']}'")
                inbatch_accuracy = 100.0
            # Note: still save the model even if the in-batch accuracy is not the best
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
            if inbatch_accuracy >= best_inbatch_accuracy:
                # if utils.is_main_process():
                #     save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
                best_inbatch_accuracy = inbatch_accuracy
                best_epoch = epoch
            log_stats = log_results(train_stats, val_status, None, epoch, best_epoch)

        if utils.is_main_process():
            # logger_out_dir = os.path.join(config.uniir_dir, config.logger_config.logger_out_dir)
            # logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
            # with open(logger_out_path, "a") as f:
            #     f.write(json.dumps(log_stats) + "\n")
            ### 仅主进程往 SwanLab 推指标，避免重复
            if config.swanlab_config.enabled:
                swanlab.log(log_stats)
        
        # Check if we should stop (signal received)
        if _checkpoint_state["should_stop"]:
            print("Stopping training due to signal...")
            break

        ### barrier：同步所有进程，确保主进程写操作完成。
	    ### empty_cache：释放未使用的显存缓存，缓解碎片问题（非必须，但长跑稳定性更好）。
        dist.barrier()  # Wait for the master process to finish writing the log file
        torch.cuda.empty_cache()


def main(config):
    is_distributed_mode = config.dist_config.distributed_mode

    # Set up seed for reproducibility
    seed = config.seed + utils.get_rank()
    set_seed(seed)

    ### 打开 cuDNN benchmark，固定输入尺寸可提升性能；若输入尺寸波动较大，会反而降低稳定性
    cudnn.benchmark = True

    # Initialize and load model
    print("Creating Jina V4 model with LoRA adapters...")
    model_config = config.model
    ckpt_config = model_config.ckpt_config
    
    if model_config.name == "JinaEmbeddingsV4Model":
        # Load the model using from_pretrained which returns a PeftModel with LoRA adapters
        # The from_pretrained method in modeling_jina_embeddings_v4.py already loads LoRA
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_config.ckpt_config.pretrained_url,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        
        # Set the task for LoRA adapter selection
        task_label = getattr(model_config, 'task_label', 'retrieval')
        model.task = task_label
        print(f"Set model task to: {task_label}")
        
    else:
        raise NotImplementedError(f"Model {config.model.name} not implemented")

    # Freeze base model, keep only LoRA trainable
    model = freeze_base_model_keep_lora(model)
    # 【新增】开启梯度检查点
    print("Enabling gradient checkpointing...")
    # 如果是 PeftModel，通常需要访问底层的 base_model 来开启
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
        
    # 这一步对于 LoRA 训练很重要，确保输入层参与梯度计算
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "enable_input_require_grads"):
        model.base_model.enable_input_require_grads()
    # Print parameter statistics before training
    if utils.is_main_process():
        print_trainable_parameters(model, "Jina Embeddings V4 (with LoRA)")

    # Set up optimizer with only trainable (LoRA) parameters
    trainer_config = config.trainer_config
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        params=trainable_params,
        lr=trainer_config.init_lr,
        weight_decay=trainer_config.weight_decay,
    )
    scaler = GradScaler()  # Initialize the GradScaler

    # Initialize checkpoint variable for later use
    checkpoint = None
    
    # If resume training, load the LoRA checkpoint
    if ckpt_config.resume_training:
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        logger.info(f"Loading LoRA checkpoint from {checkpoint_path}")
        model, checkpoint = load_lora_checkpoint(model, checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    # Move model to GPUs
    model.train()
    model = model.to(config.dist_config.gpu_id)
    model_without_ddp = model
    if is_distributed_mode:
        # find_unused_parameters=True is needed for PEFT models with some frozen params
        model = DDP(model, device_ids=[config.dist_config.gpu_id], find_unused_parameters=True)
        model_without_ddp = model.module

    # Prepare datasets and dataloaders
    logger.info("Preparing dataset ...")  # Note printing only available in the main process
    logger.info(f"Loading dataset from {config.mbeir_data_dir}{config.data_config.train_query_data_path}...")

    # Get preprocessing functions from the base model
    # For PeftModel, we need to access the underlying model
    if hasattr(model_without_ddp, 'base_model'):
        base_model_for_preprocess = model_without_ddp.base_model.model
    else:
        base_model_for_preprocess = model_without_ddp
    
    img_preprocess_fn = base_model_for_preprocess.get_img_preprocess_fn()
    tokenizer = base_model_for_preprocess.get_tokenizer()
    
    ### 获取全局并行度与当前 rank
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    ### 数据集核心构建函数
    train_dataset, train_collector = build_mbeir_dataset_from_config(
        config=config,
        tokenizer=tokenizer,
        img_preprocess_fn=img_preprocess_fn,
        dataset_type=DatasetType.MAIN_TRAIN,
    )
    ### DDP 必备：按 rank 均匀切分数据子集并控制 shuffle 同步
    ### 注意：设置 drop_last=True 确保所有 rank 有相同数量的样本，避免 NCCL 超时
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
        drop_last=True,  # 确保所有 rank 有相同数量的 batch，避免 synchronize 时超时
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader_config.train_batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,  # Note: since we use sampler, shuffle should be False
        collate_fn=train_collector,
        drop_last=True,
    )

    enable_eval = config.evaluator.enable_eval
    valid_loader = None
    if enable_eval:
        in_batch_val_dataset, in_batch_val_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.IN_BATCH_VAL,
        )
        in_batch_val_sampler = DistributedSampler(
            dataset=in_batch_val_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
            drop_last=True,  # 确保所有 rank 有相同数量的 batch，避免 synchronize 时超时
        )
        valid_loader = DataLoader(
            dataset=in_batch_val_dataset,
            batch_size=config.dataloader_config.valid_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=True,
            sampler=in_batch_val_sampler,
            shuffle=False,  # Note: since we use sampler, shuffle should be False
            collate_fn=in_batch_val_collector,
            drop_last=True,
        )
    else:
        print("In-batch validation is disabled.")

    # Initializing the scheduler
    ### 计算总的 调度步数：
	### •	这里用的是 步数级 退火（而非 epoch 级），即 T_max = 总迭代步数（考虑了梯度累积）。
    ### 可加入 warmup（如 GradualWarmupScheduler 或 get_cosine_schedule_with_warmup），对大模型更稳。
    t_total = (
        len(train_loader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=0)

    epoch = 0

    # Resume training: load scheduler state and epoch
    if ckpt_config.resume_training and checkpoint is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {epoch}")

    # Training loop
    ### barrier 一次，确保各 rank 均准备好
    dist.barrier()
    train(
        train_loader,
        valid_loader,
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--uniir_dir",
        type=str,
        default="/data/UniIR",
        help="Path to mbeir directory to save checkpoints, embeddings, etc.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data",
        help="Path to mbeir dataset directory",
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    ### 使用 OmegaConf 读取 YAML 到 DictConfig
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    # Initialize distributed training
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    ### utils.init_distributed_mode(args) 内部会解析 RANK/LOCAL_RANK/WORLD_SIZE 等环境变量或启动参数，设置 args.gpu、args.distributed。
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    # Set up SwanLab
    if config.swanlab_config.enabled and utils.is_main_process():
        ### 从 .env 读取 SWANLAB_API_KEY (可选)
        load_dotenv()
        swanlab_key = os.environ.get("SWANLAB_API_KEY")
        swanlab_project = getattr(config.swanlab_config, 'project', 'jina-v4-training')
        
        # SwanLab 支持离线模式，API key 可选
        if swanlab_key:
            swanlab.login(api_key=swanlab_key)
        
        swanlab.init(
            project=swanlab_project,
            experiment_name=config.swanlab_config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
            mode=getattr(config.swanlab_config, 'mode', 'cloud'),  # 'cloud' or 'local'
        )

    # Set up logger
    if utils.is_main_process():
        logger_out_dir = os.path.join(config.uniir_dir, config.logger_config.logger_out_dir)
        logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
        if not os.path.exists(logger_out_dir):
            os.makedirs(logger_out_dir, exist_ok=True)
        ### 同时写文件与控制台
        handlers = [logging.FileHandler(logger_out_path), logging.StreamHandler()]
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            level=logging.DEBUG,
            datefmt="%d-%m-%Y %H:%M:%S",
            handlers=handlers,
        )
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logger = logging.getLogger(__name__)
        logger.info(config)

    main(config)

    # Close swanlab
    if config.swanlab_config.enabled and utils.is_main_process():
        swanlab.finish()

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
