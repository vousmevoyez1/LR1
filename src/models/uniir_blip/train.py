# Standard library
import argparse
import logging
import os
import random
import time
import datetime

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
import wandb

# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_distributed_sampler_list,
    build_dataloader_list,
)
from models.uniir_blip.backbone.blip import load_checkpoint
from models.uniir_blip.blip_featurefusion.blip_ff import blip_ff
from models.uniir_blip.blip_scorefusion.blip_sf import blip_sf
from models.uniir_blip.engine import train_one_epoch, eval_engine
import models.uniir_blip.utils as utils

# Set up logger
logger = logging.getLogger()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config):
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    save_obj = {
        "model": model.state_dict(),
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
    ### best_inbatch_accuracy 用于追踪最优 in-batch acc。
    global_step, total_loss, best_inbatch_accuracy = (
        0,
        0.0,
        0.0,
    )  # TODO: global_step is not used.
    best_epoch = 0
    model.zero_grad()

    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        # Set different seed for different epoch
        ### DDP 关键点：每个 epoch 为 DistributedSampler 设置新种子，确保各 rank 的 shuffle 一致。
        if is_distributed_mode:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
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
            ### 仅主进程往 WandB 推指标，避免重复
            if config.wandb_config.enabled:
                wandb.log(log_stats)

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

    #### 换成Jina V4
    # Initialize and load model
    print("Creating BLIP model...")
    model_config = config.model
    ckpt_config = model_config.ckpt_config
    if model_config.name == "BLIPScoreFusion":
        model = blip_sf(
            pretrained=ckpt_config.pretrained_blip_url,  # This always saved to cache
            image_size=model_config.image_size,
            vit=model_config.vit,
            vit_grad_ckpt=model_config.vit_grad_ckpt,
            vit_ckpt_layer=model_config.vit_ckpt_layer,
            embed_dim=model_config.embed_dim,
            queue_size=model_config.queue_size,
            config=model_config,
        )
    elif model_config.name == "BLIPFeatureFusion":
        model = blip_ff(
            pretrained=ckpt_config.pretrained_blip_url,  # This always saved to cache
            ### 224
            image_size=model_config.image_size,
            ### large
            vit=model_config.vit,
            ### True，开启梯度检查点，显著节省显存（以算力换显存），训练略变慢。
            vit_grad_ckpt=model_config.vit_grad_ckpt,
            ### 对 ViT 前 12 层或指定段落启用检查点（具体取决于实现）。可微调该值来平衡显存/速度。
            vit_ckpt_layer=model_config.vit_ckpt_layer,
            ### 768
            embed_dim=model_config.embed_dim,
            ### 57960
            queue_size=model_config.queue_size,
            config=model_config,
        )
    else:
        raise NotImplementedError(f"Model {config.model} not implemented")

    # Set up optimizer, and scaler
    trainer_config = config.trainer_config
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=trainer_config.init_lr,
        weight_decay=trainer_config.weight_decay,
    )
    scaler = GradScaler()  # Initialize the GradScaler

    #### 换了模型，需要考虑修改这里的代码
    # If resume training, load the checkpoint
    if ckpt_config.resume_training:
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        logger.info(f"loading {config.model.name} checkpoint from {checkpoint_path}")
        model, msg = load_checkpoint(model, checkpoint_path)
        print("missing keys:")
        print(msg.missing_keys)
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    # Move model to GPUs
    model.train()
    model = model.to(config.dist_config.gpu_id)
    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id])
        model_without_ddp = model.module

    # Prepare datasets and dataloaders
    logger.info("Preparing dataset ...")  # Note printing only available in the main process
    logger.info(f"Loading dataset from {config.mbeir_data_dir}{config.data_config.train_query_data_path}...")

    #### dataset关于img_preprocess_fn和tokenizer需要与jina v4对上，img_preprocess_fn应该不用管，就lamda (x:x)就行，需要考虑tokenizer在collator的使用
    ### 从模型暴露的接口获取图像预处理函数与分词器，保证与模型要求完全一致（避免数据/模型不一致）。
    img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    tokenizer = model_without_ddp.get_tokenizer()
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
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
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
    if ckpt_config.resume_training:
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1

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

    # Set up wandb
    if config.wandb_config.enabled and utils.is_main_process():
        ### 从 .env 读取 WANDB_API_KEY/PROJECT/ENTITY
        load_dotenv()  # Load .env and get WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY
        wandb_key = os.environ.get("WANDB_API_KEY")
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_entity = os.environ.get("WANDB_ENTITY")

        if not wandb_key:
            raise ValueError("WANDB_API_KEY not found. Ensure it's set in the .env file.")

        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=config.wandb_config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
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

    # Close wandb
    if config.wandb_config.enabled and utils.is_main_process():
        wandb.finish()

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
