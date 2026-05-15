# Standard library
import argparse
import logging
import os
import random
import signal
import sys

# Third-party
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
# Local
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
)

from models.qwen3.engine import train_one_epoch, eval_engine
import models.qwen3.utils as utils
from models.qwen3.qwen3_thought_wrapper import Qwen3VLThoughtWrapper
from models.qwen3.utils import freeze_base_model_keep_lora


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------------

def _rotate_step_checkpoints(ckpt_dir: str, model_name: str, save_total_limit: int):
    """Remove old step checkpoints, keeping only the most recent ones."""
    import glob

    pattern = os.path.join(ckpt_dir, f"{model_name}_step_*.pth")
    step_ckpts = glob.glob(pattern)

    if len(step_ckpts) <= save_total_limit:
        return

    def get_step(path: str) -> int:
        basename = os.path.basename(path)
        try:
            return int(basename.split('_step_')[1].replace('.pth', ''))
        except Exception:
            return 0

    step_ckpts.sort(key=get_step)
    for old_ckpt in step_ckpts[:-save_total_limit]:
        try:
            os.remove(old_ckpt)
            print(f"[Rotation] Removed old checkpoint: {old_ckpt}")
        except Exception as e:
            print(f"[Rotation] Failed to remove {old_ckpt}: {e}")


def _collect_trainable_state_dict(model_to_save):
    """Collect trainable params only (LoRA + thought/final token embeddings)."""
    trainable = {}
    for name, param in model_to_save.named_parameters():
        if param.requires_grad:
            trainable[name] = param.detach().cpu()
    return trainable


def save_step_checkpoint(model, optimizer, scheduler, epoch, global_step, scaler, config):
    """Save step-level checkpoint for resuming mid-epoch training."""
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_step_{global_step}.pth"

    model_to_save = model.module if hasattr(model, "module") else model

    save_obj = {
        "trainable_state_dict": _collect_trainable_state_dict(model_to_save),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }

    ckpt_dir = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, checkpoint_name)
    torch.save(save_obj, checkpoint_path)
    print(f"[Step {global_step}] Saved checkpoint to {checkpoint_path}")

    save_total_limit = getattr(config.trainer_config, "save_total_limit", 0)
    if save_total_limit and save_total_limit > 0:
        _rotate_step_checkpoints(ckpt_dir, model_name, save_total_limit)


def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config):
    """Save epoch-level checkpoint."""
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"

    model_to_save = model.module if hasattr(model, "module") else model

    save_obj = {
        "trainable_state_dict": _collect_trainable_state_dict(model_to_save),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }

    checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint_trainable(model, checkpoint_path: str):
    """Load trainable parameters from our checkpoint format."""
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"Checkpoint file {checkpoint_path} does not exist")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "trainable_state_dict" not in checkpoint:
        raise RuntimeError(f"Checkpoint {checkpoint_path} missing 'trainable_state_dict'")

    state = checkpoint["trainable_state_dict"]
    model_state = model.state_dict()

    loaded = 0
    missing = 0
    for name, tensor in state.items():
        if name in model_state:
            model_state[name].copy_(tensor)
            loaded += 1
        else:
            missing += 1

    print(f"Loaded trainable params from {checkpoint_path}: {loaded} loaded, {missing} missing")
    return model, checkpoint


# -----------------------------------------------------------------------------
# Signal handling (graceful stop)
# -----------------------------------------------------------------------------

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
    signal_name = signal.Signals(signum).name
    print(f"\n{'='*60}")
    print(f"Received {signal_name} signal. Will stop after saving a checkpoint...")
    print(f"{'='*60}")

    _checkpoint_state["should_stop"] = True

    if utils.is_main_process() and _checkpoint_state["model"] is not None:
        # Save a step checkpoint at the current global_step.
        try:
            save_step_checkpoint(
                _checkpoint_state["model"],
                _checkpoint_state["optimizer"],
                _checkpoint_state["scheduler"],
                _checkpoint_state["epoch"],
                _checkpoint_state["global_step"],
                _checkpoint_state["scaler"],
                _checkpoint_state["config"],
            )
        except Exception as e:
            print(f"[EMERGENCY] Failed to save checkpoint: {e}")

    if dist.is_initialized():
        dist.destroy_process_group()

    sys.exit(0)


# -----------------------------------------------------------------------------
# Training loop wrapper
# -----------------------------------------------------------------------------

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

    if utils.is_main_process():
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        print("Registered signal handlers for graceful checkpoint saving")

    _checkpoint_state["model"] = model_without_ddp
    _checkpoint_state["optimizer"] = optimizer
    _checkpoint_state["scheduler"] = scheduler
    _checkpoint_state["scaler"] = scaler
    _checkpoint_state["config"] = config

    global_step = 0
    
    # Get step-level evaluation frequency
    eval_steps = getattr(config.evaluator, 'eval_steps', 0)
    if eval_steps == 0:
        eval_steps = getattr(config.trainer_config, 'eval_steps', 0)
    
    if utils.is_main_process() and eval_steps > 0:
        print(f"[INFO] Step-level validation enabled: every {eval_steps} steps")

    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        _checkpoint_state["epoch"] = epoch

        if is_distributed_mode:
            train_loader.sampler.set_epoch(epoch)

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
            val_loader=val_loader,
            eval_steps=eval_steps,
            model_without_ddp=model_without_ddp,
        )

        if isinstance(train_result, tuple):
            train_stats, global_step = train_result
        else:
            train_stats = train_result
            steps_per_epoch = len(train_loader) // config.trainer_config.gradient_accumulation_steps
            global_step = (epoch + 1) * steps_per_epoch

        _checkpoint_state["global_step"] = global_step

        eval_freq = config.evaluator.eval_freq
        if val_loader is None or epoch % eval_freq != 0:
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
        else:
            val_stats = eval_engine(model_without_ddp, model, val_loader, gpu_id, config)
            if utils.is_main_process():
                # Log validation metrics to console
                print(f"\n{'='*60}")
                print(f"Validation Results (Epoch {epoch}):")
                for key, value in val_stats.items():
                    print(f"  {key}: {value:.6f}")
                print(f"{'='*60}\n")
                
                # Log validation metrics to SwanLab if enabled
                if config.get("swanlab_config", {}).get("enabled", False):
                    import swanlab
                    val_metrics = {f"val/{key}": value for key, value in val_stats.items()}
                    val_metrics["epoch"] = epoch
                    swanlab.log(val_metrics, step=global_step)
                
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)

        if _checkpoint_state["should_stop"]:
            print("Stopping training due to signal...")
            break

        # Removed redundant barrier - train_one_epoch already synchronizes at the end
        torch.cuda.empty_cache()


def main(config):
    is_distributed_mode = config.dist_config.distributed_mode

    seed = config.seed + utils.get_rank()
    set_seed(seed)
    cudnn.benchmark = True
    # ================== [新增] 初始化 SwanLab ==================
    if utils.is_main_process() and config.get("swanlab_config", {}).get("enabled", False):
        import swanlab
        from omegaconf import OmegaConf
        
        # 将 OmegaConf 转为普通字典，解析插值变量 (如 ${experiment.description})
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        swanlab.init(
            project=config.swanlab_config.project,
            name=config.swanlab_config.experiment_name,
            mode=config.swanlab_config.mode,
            config=config_dict,
        )
        print("🚀 SwanLab 云端可视化看板已初始化！")
    # ==========================================================
    model_config = config.model
    ckpt_config = model_config.ckpt_config

    # Initialize model
    model = Qwen3VLThoughtWrapper(
        model_name_or_path=model_config.original_model_name,
        max_length=getattr(model_config, "mbeir_max_text_length", 1500),
        torch_dtype=torch.bfloat16,
        attn_implementation=getattr(model_config, "attn_implementation", "sdpa"),
    )
    model.debug_mode = bool(
        getattr(model_config, "debug_mode", False)
        or getattr(config.trainer_config, "debug_mode", False)
    )
    model.debug_max_text_chars = int(getattr(config.trainer_config, "debug_max_text_chars", 120))
    model.debug_preprocess_print_freq = int(getattr(config.trainer_config, "debug_print_freq", 1))
    model.debug_token_preview_len = int(getattr(config.trainer_config, "debug_token_preview_len", 12))
    model.mbeir_symmetric_encoding = bool(
        getattr(model_config, "symmetric_query_candidate_encoding", False)
    )
    if utils.is_main_process():
        print(
            "[Encoding Mode] symmetric_query_candidate_encoding="
            f"{model.mbeir_symmetric_encoding}"
        )

    # Memory optimizations to allow larger batch size
    if getattr(model_config, "gradient_checkpointing", True):
        try:
            model.model.gradient_checkpointing_enable()
            # Keep dropout/checkpointing behavior consistent
            if hasattr(model.model.config, "use_cache"):
                model.model.config.use_cache = False
            if hasattr(model.model.generation_config, "use_cache"):
                model.model.generation_config.use_cache = False
            print("Enabled gradient checkpointing and disabled KV cache")
        except Exception as e:
            print(f"[Warn] Failed to enable gradient checkpointing: {e}")
    # ===== [新增] 注入 LoRA 配置 =====
    print("Injecting LoRA adapters into the language model...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    # 给内部的语言模型打上 LoRA 补丁
    model.model.language_model = get_peft_model(model.model.language_model, lora_config)
    # =================================
    # Setup thought tokens before freezing (so token ids exist)
    num_thought_tokens = getattr(model_config, "num_thought_tokens", 0)
    enable_final_token = getattr(model_config, "enable_final_token", True)
    if (num_thought_tokens and num_thought_tokens > 0) or enable_final_token:
        semantic_init_token = getattr(model_config, "semantic_init_token", ".")
        skip_init = getattr(model_config, "skip_thought_init", False)
        model.setup_thought_tokens(
            num_thought_tokens,
            semantic_init_token=semantic_init_token,
            skip_init=skip_init,
            enable_final_token=enable_final_token,
        )

    thought_token_ids = None
    thought_ids = getattr(model, "_thought_token_ids", None)
    final_id = getattr(model, "_final_token_id", None)
    if thought_ids is not None:
        thought_token_ids = list(thought_ids)
        if final_id is not None:
            thought_token_ids.append(final_id)
        if len(thought_token_ids) == 0:
            thought_token_ids = None

    # Freeze base model, keep LoRA trainable. Special-token embeddings are handled
    # by ReasoningTokenEmbeddingWrapper (small trainable table) inside the wrapper.
    freeze_base_model_keep_lora(model, thought_token_ids=None)

    # Make sure special token embeddings are actually saved/trained.
    # We train only the small special embedding table, NOT the full vocab embedding.
    if thought_token_ids is not None and len(thought_token_ids) > 0:
        embed_layer = model.model.get_input_embeddings()
        if not hasattr(embed_layer, "special_embedding"):
            raise RuntimeError(
                "Expected ReasoningTokenEmbeddingWrapper as input embeddings, "
                "but got a different embedding module."
            )
        embed_layer.special_embedding.weight.requires_grad = True
    trainer_config = config.trainer_config
    # 分组设置不同学习率：特殊 Token Embedding (较高 LR)，LoRA 参数 (正常 LR)
    special_params = []
    lora_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "special_embedding" in name:
                special_params.append(p)
            else:
                lora_params.append(p)

    special_lr = getattr(trainer_config, "special_token_lr", trainer_config.init_lr * 10) # 默认10倍
    
    optimizer = torch.optim.AdamW(
        [
            {"params": special_params, "lr": special_lr},
            {"params": lora_params, "lr": trainer_config.init_lr}
        ],
        weight_decay=trainer_config.weight_decay,
    )

    scaler = GradScaler("cuda")

    checkpoint = None
    epoch = 0

    if ckpt_config.resume_training:
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model, checkpoint = load_checkpoint_trainable(model, checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    # Move model to GPU
    model.train()
    model = model.to(config.dist_config.gpu_id)
    model_without_ddp = model

    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id], find_unused_parameters=True, broadcast_buffers=False)
        # [核心修复] 告诉 DDP 锁定计算图，无视多次 forward 和 Checkpoint 带来的重复 Hook 触发
        model._set_static_graph()
        model_without_ddp = model.module

    # Dataset / dataloader
    img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    tokenizer = model_without_ddp.get_tokenizer()

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    train_dataset, train_collector = build_mbeir_dataset_from_config(
        config=config,
        tokenizer=tokenizer,
        img_preprocess_fn=img_preprocess_fn,
        dataset_type=DatasetType.MAIN_TRAIN,
    )

    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader_config.train_batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=train_collector,
        drop_last=True,
    )

    valid_loader = None
    if config.evaluator.enable_eval:
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
            drop_last=True,
        )

        valid_loader = DataLoader(
            dataset=in_batch_val_dataset,
            batch_size=config.dataloader_config.valid_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=True,
            sampler=in_batch_val_sampler,
            shuffle=False,
            collate_fn=in_batch_val_collector,
            drop_last=True,
        )

    # Scheduler: step-based cosine annealing
    t_total = (
        len(train_loader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=0)

    if ckpt_config.resume_training and checkpoint is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {epoch}")

    # NOTE: avoid barrier here; it can deadlock if any rank exits early.
    # DDP will synchronize on the first collective op during training anyway.

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
    config = OmegaConf.load(args.config_path)

    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    # Init distributed
    args.dist_url = config.dist_config.dist_url
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    if utils.is_main_process():
        handlers = [logging.StreamHandler()]
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            level=logging.INFO,
            datefmt="%d-%m-%Y %H:%M:%S",
            handlers=handlers,
        )
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logger.info(config)

    main(config)

    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
