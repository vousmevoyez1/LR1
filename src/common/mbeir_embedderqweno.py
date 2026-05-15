"""
This module generates embeddings for MBEIR with multiple GPUs.
"""

import os
import argparse
from omegaconf import OmegaConf
import tqdm
import gc
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
# from torch.cuda.amp import autocast
from torch.amp import autocast
import transformers


def _ensure_qwen_vl_utils_available():
    """Make qwen_vl_utils importable without polluting torch import."""
    try:
        import qwen_vl_utils  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    # Qwen3-VL-Embedding repo sometimes vendors this inside its own venv.
    cand_site = "/data/Qwen3-VL-Embedding/.venv/lib/python3.12/site-packages"
    if os.path.isdir(os.path.join(cand_site, "qwen_vl_utils")):
        import sys

        # Append (not prepend) to avoid shadowing current environment packages.
        sys.path.append(cand_site)

    # Re-try import (will raise if still missing)
    import qwen_vl_utils  # noqa: F401

import dist_utils
from dist_utils import ContiguousDistributedSampler
from utils import build_model_from_config, set_seed
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    Mode,
)


@torch.no_grad()
def generate_embeds_and_ids_for_dataset_with_gather(model, data_loader, device, use_fp16=True):
    embedding_tensors = []
    id_list = []

    total_cores = os.cpu_count()
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        initial_threads_per_process = 2
        torch.set_num_threads(initial_threads_per_process)
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank}")
    for batch in data_loader:
        # Used in combination with pin_memory=True
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(device)
        # Enable autocast
        with autocast(device_type="cuda", enabled=use_fp16, dtype=torch.bfloat16):
            embeddings_batched, ids_list_batched = model(encode_mbeir_batch=True, **batch)

        ### embeddings_batched.shape [batch_size, 2048] len(ids_list_batched) = batch_size
        ###  print(f"Batch embeddings shape: {embeddings_batched.shape}, Batch ids count: {len(ids_list_batched)}")
        
        embedding_tensors.append(embeddings_batched.half())  # We only save FP16 embeddings to save space.
        id_list.extend(ids_list_batched)

    # Convert list of tensors to a single tensor
    embedding_tensor = torch.cat(embedding_tensors, dim=0)
    embedding_list = None

    if dist.is_initialized():
        # First, share the sizes of tensors across processes
        size_tensor = torch.tensor([embedding_tensor.size(0)], dtype=torch.long, device=device)

        # Allocate tensors to gather results on rank 0
        if dist.get_rank() == 0:
            # Allocate tensors with the correct sizes on rank 0
            gathered_embeddings = [torch.empty_like(embedding_tensor) for _ in range(dist.get_world_size())]
            id_list_gathered = [list() for _ in range(dist.get_world_size())]
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
        else:
            gathered_embeddings = None
            id_list_gathered = None
            sizes = None

        # Synchronize all processes before gathering data
        dist.barrier()

        # Gather embeddings from all processes
        dist.gather(embedding_tensor, gather_list=gathered_embeddings, dst=0)
        # Gather ids from all processes
        dist.gather_object(id_list, object_gather_list=id_list_gathered, dst=0)
        # Gather sizes from all processes
        dist.gather(size_tensor, gather_list=sizes, dst=0)

        # Synchronize all processes after gathering data
        dist.barrier()

        # On the main process, concatenate the results
        if dist.get_rank() == 0:
            print(f"Embedder Log: Gathered embeddings and ids on rank 0, starting to process them...")

            # Increase number of threads for rank 0 during conversion
            torch.set_num_threads(8)

            # Trim the last tensors based on the gathered sizes
            gathered_embeddings[-1] = gathered_embeddings[-1][: sizes[-1][0]]
            embedding_list = torch.cat(gathered_embeddings, dim=0)

            # Flatten gathered ids
            id_list = [id for sublist in id_list_gathered for id in sublist]
            assert len(id_list) == embedding_list.size(0)
            # Check unique ids
            assert len(id_list) == len(set(id_list)), "Hashed IDs should be unique"
            print(f"Embedder Log: Finished processing embeddings and ids on rank 0.")

            # Note: we are using float16 to save space, and the precision loss is negligible.
            embedding_list = embedding_list.half().cpu().numpy()
            print(f"Embedder Log: Converted embedding_list to cpu numpy array of type {embedding_list.dtype}.")

            # Reset number of threads to initial value after conversion
            torch.set_num_threads(8)

        dist.barrier()  # Wait for rank 0 to process the embeddings and ids.
    else:
        embedding_list = embedding_tensor.half().cpu().numpy()

    return embedding_list, id_list


@torch.no_grad()
def generate_embeds_and_ids_for_dataset_with_tmp_files(
    model, data_loader, embed_dir, dataset_name, split_name, device, use_fp16=True
):
    embedding_tensors = []
    hashed_id_list = []

    if dist.is_initialized():
        rank = dist.get_rank()
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank}")

    for batch in data_loader:
        # Used in combination with pin_memory=True
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Enable autocast
        with autocast(device_type="cuda", enabled=use_fp16, dtype=torch.bfloat16):
            embeddings_batched, hashed_ids_list_batched = model(batch, encode_mbeir_batch=True)

        embedding_tensors.append(embeddings_batched)
        hashed_id_list.extend(hashed_ids_list_batched)

    # Convert list of tensors to a single tensor
    embedding_tensor = torch.cat(embedding_tensors, dim=0)
    assert embedding_tensor.size(0) == len(hashed_id_list), "embeddings and ids must have the same size."

    # Save the embeddings and ids to .npy on each GPU
    all_embeddings = []
    all_ids = []
    if dist.is_initialized():
        gpu_id = device
        embed_tmp_data_name = f"mbeir_{dataset_name}_{split_name}_embed_gpu_{gpu_id}.npy"
        embed_tmp_id_name = f"mbeir_{dataset_name}_{split_name}_ids_gpu_{gpu_id}.npy"
        embed_tmp_data_path = os.path.join(embed_dir, embed_tmp_data_name)
        embed_tmp_id_path = os.path.join(embed_dir, embed_tmp_id_name)

        # Note: we are using float16 to save space, and the precision loss is negligible.
        np.save(embed_tmp_data_path, embedding_tensor.half().cpu().numpy())
        np.save(embed_tmp_id_path, hashed_id_list)

        print(f"Saved embeddings to {embed_tmp_data_path} and ids to {embed_tmp_id_path}.")
        dist.barrier()  # Ensure every process has finished saving.

        if dist.get_rank() == 0:
            print("Loading tmp embeddings and ids on rank 0...")

            for gpu_id in range(dist.get_world_size()):
                embed_data_name = f"mbeir_{dataset_name}_{split_name}_embed_gpu_{gpu_id}.npy"
                embed_id_name = f"mbeir_{dataset_name}_{split_name}_ids_gpu_{gpu_id}.npy"  # Loading IDs using .npy

                embed_path = os.path.join(embed_dir, embed_data_name)
                id_path = os.path.join(embed_dir, embed_id_name)

                embeddings = np.load(embed_path)
                all_embeddings.append(embeddings)

                ids = np.load(id_path)
                all_ids.extend(ids)

            all_embeddings = np.concatenate(all_embeddings, axis=0)
            assert len(all_embeddings) == len(all_ids), "Mismatch between embeddings and IDs length."
            print(f"Finished processing tmp embeddings and ids on rank 0.")

        dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.
    else:
        all_embeddings = embedding_tensor.half().cpu().numpy()
        all_ids = hashed_id_list

    return all_embeddings, all_ids


@torch.no_grad()
def generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config):
    """This script generates embeddings for the queries and candidates(doc)"""
    uniir_dir = config.uniir_dir
    mbeir_data_dir = config.mbeir_data_dir
    embed_config = config.embed_config
    embed_dir_name = embed_config.embed_dir_name
    expt_dir_name = config.experiment.path_suffix

    if not expt_dir_name.endswith("/"):
        expt_dir_name = expt_dir_name + "/"
        config.experiment.path_suffix = expt_dir_name
        if dist_utils.is_main_process():
            print(f"[Warn] Normalized experiment.path_suffix to: {expt_dir_name}")

    # Config for dataset
    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))

    splits = []
    # Load the dataset splits to embed
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        ### query/test
        split_dir_name = getattr(data_config, f"{split_name}_dir_name")
        embed_dataset_config = getattr(embed_config, f"{split_name}_datasets_config", None)
        ### only embed if enabled (now is test split)
        if embed_dataset_config and embed_dataset_config.enable_embed:
            dataset_name_list = getattr(embed_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(embed_dataset_config, "correspond_cand_pools_name", None)
            splits.append((split_name, split_dir_name, dataset_name_list, cand_pool_name_list))
            assert len(dataset_name_list) == len(cand_pool_name_list), "Mismatch between datasets and candidate pools."

    # Load the candidate pool to embed
    embed_cand_pool_config = embed_config.cand_pools_config
    if embed_cand_pool_config and embed_cand_pool_config.enable_embed:
        split_name = "cand_pool"
        split_dir_name = data_config.cand_pool_dir_name
        cand_pool_name_list = embed_cand_pool_config.cand_pools_name_to_embed
        splits.append(
            (
                split_name,
                split_dir_name,
                [None] * len(cand_pool_name_list),
                cand_pool_name_list,
            )
        )

    # Pretty Print dataset and candidate pool to embed
    if dist_utils.is_main_process():
        print("-" * 30)
        for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
            if split_name == "cand_pool":
                print(f"Split: {split_name}, Split dir: {split_dir}, Candidate pools to embed: {cand_pool_name_list}")
            else:
                print(f"Split: {split_name}, Split dir: {split_dir}, Datasets to embed: {dataset_name_list}")
        print("-" * 30)

    # Generate embeddings
    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            # Determine output file paths to check if already embedded
            if split_name == "cand_pool":
                mid_name = cand_pool_name.lower()
            else:
                mid_name = dataset_name.lower() if dataset_name else None
            
            if mid_name:
                embed_data_name = f"mbeir_{mid_name}_{split_name}_embed.npy"
                id_data_name = f"mbeir_{mid_name}_{split_name}_ids.npy"
                embed_path = os.path.join(
                    uniir_dir,
                    embed_dir_name,
                    expt_dir_name,
                    split_name,
                    embed_data_name,
                )
                id_path = os.path.join(
                    uniir_dir,
                    embed_dir_name,
                    expt_dir_name,
                    split_name,
                    id_data_name,
                )

                overwrite = bool(getattr(embed_config, "overwrite", False))
                skip_if_exists = not overwrite

                # If a fine-tuned checkpoint is provided, we must re-embed even if files exist.
                # Otherwise retrieval may silently use stale (pre-finetune) embeddings.
                if skip_if_exists and os.path.exists(embed_path) and os.path.exists(id_path):
                    if dist_utils.is_main_process():
                        print(f"\n[SKIP] Embeddings already exist:")
                        print(f"       - {embed_path}")
                        print(f"       - {id_path}")
                        print(f"       Skipping {mid_name} in {split_name} split.")
                    if dist.is_initialized():
                        dist.barrier()  # Sync all processes
                    continue
                elif overwrite and dist_utils.is_main_process() and os.path.exists(embed_path) and os.path.exists(id_path):
                    print(f"\n[OVERWRITE] Re-generating embeddings:")
                    print(f"            - {embed_path}")
                    print(f"            - {id_path}")
            
            if split_name == "cand_pool":
                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_{split_name}.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\nEmbedder Log: Generating embeddings for {cand_pool_data_path}...")
                    print_config = True
                # TODO: refactor this and use build dataset from Config.
                dataset = MBEIRCandidatePoolDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    cand_pool_data_path=cand_pool_data_path,
                    img_preprocess_fn=img_preprocess_fn,
                    print_config=print_config,
                )
                collator = MBEIRCandidatePoolCollator(
                    tokenizer=tokenizer,
                    image_size=image_size,
                )
            else:  # "train" or "val" or "test"
                # Construct query data path
                dataset_name = dataset_name.lower()
                query_data_name = f"mbeir_{dataset_name}_{split_name}.jsonl"
                query_data_path = os.path.join(split_dir, query_data_name)

                # Construct the candidate pool path
                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\nEmbedder Log: Generating embeddings for {query_data_path} with {cand_pool_data_path}...")
                    print_config = True
                mode = Mode.EVAL
                dataset = MBEIRMainDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    query_data_path=query_data_path,
                    cand_pool_path=cand_pool_data_path,
                    query_instruct_path=query_instruct_path,
                    img_preprocess_fn=img_preprocess_fn,
                    mode=mode,
                    enable_query_instruct=data_config.enable_query_instruct,
                    shuffle_cand=data_config.shuffle_cand,
                    print_config=print_config,
                )
                collator = MBEIRMainCollator(
                    tokenizer=tokenizer,
                    image_size=image_size,
                    mode=mode,
                )

            # Config for data loader
            batch_size = config.dataloader_config.batch_size
            num_workers = config.dataloader_config.num_workers

            # Set up distributed data parallel
            num_tasks = dist_utils.get_world_size()
            global_rank = dist_utils.get_rank()
            sampler = ContiguousDistributedSampler(
                dataset,
                num_replicas=num_tasks,
                rank=global_rank,
            )  # Note: assume the dataset is in sorted order.
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=False,  # Since we have distributed sampler, we don't need to shuffle the data here.
                collate_fn=collator,
                drop_last=False,
            )
            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.
            if dist_utils.is_main_process():
                if split_name == "cand_pool":
                    print(f"Embedder Log: Data loader for {cand_pool_data_path} is set up.")
                    print(f"Embedder Log: Generating embeddings for {cand_pool_data_path}...")
                else:
                    print(
                        f"Embedder Log: Data loader for {query_data_path} with candidate pool {cand_pool_data_path} is set up."
                    )
                    print(f"Embedder Log: Generating embeddings for {query_data_path} ...")
                print(f"Inference with half precision: {config.embed_config.use_fp16}")

            # Generate embeddings and ids
            embedding_list, id_list = generate_embeds_and_ids_for_dataset_with_gather(
                model,
                data_loader,
                device=config.dist_config.gpu_id,
                use_fp16=config.embed_config.use_fp16,
            )

            # Save the embeddings and ids to .npy
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Embedder Log: Embedding list length: {len(embedding_list)}")
                print(f"Embedder Log: ID list length: {len(id_list)}")

                mid_name = cand_pool_name if split_name == "cand_pool" else dataset_name
                # Save the embeddings to .npy
                embed_data_name = f"mbeir_{mid_name}_{split_name}_embed.npy"
                embed_path = os.path.join(
                    uniir_dir,
                    embed_dir_name,
                    expt_dir_name,
                    split_name,
                    embed_data_name,
                )
                os.makedirs(os.path.dirname(embed_path), exist_ok=True)
                np.save(embed_path, embedding_list)
                print(f"Embedder Log: Saved embeddings to {embed_path}.")

                # Save the IDs to .npy
                id_data_name = f"mbeir_{mid_name}_{split_name}_ids.npy"
                id_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, id_list)
                print(f"Embedder Log: Saved ids to {id_path}.")

            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.

            # Delete the embeddings and IDs to free up memory
            del embedding_list
            del id_list
            del data_loader
            del dataset
            del collator
            del sampler

            # Explicitly call the garbage collector
            gc.collect()
            torch.cuda.empty_cache()

        # Union pool embeddings
        if split_name == "cand_pool" and embed_cand_pool_config.embed_union_pool:
            # Check if union pool embeddings already exist
            union_embed_path = os.path.join(
                uniir_dir,
                embed_dir_name,
                expt_dir_name,
                split_name,
                f"mbeir_union_{split_name}_embed.npy",
            )
            union_id_path = os.path.join(
                uniir_dir,
                embed_dir_name,
                expt_dir_name,
                split_name,
                f"mbeir_union_{split_name}_ids.npy",
            )
            
            if os.path.exists(union_embed_path) and os.path.exists(union_id_path):
                if dist_utils.is_main_process():
                    print(f"\n[SKIP] Union pool embeddings already exist:")
                    print(f"       - {union_embed_path}")
                    print(f"       - {union_id_path}")
                    print(f"       Skipping union pool generation.")
                if dist.is_initialized():
                    dist.barrier()
                continue
            
            # To efficiently generate embeddings for the union(global) pool,
            # We concat previously saved embeddings and ids from single(local) pool
            # Instead of embed the union pool directly.
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\nEmbedder Log: Generating embeddings for union pool...")

                # Increase number of threads for rank 0
                world_size = dist.get_world_size()
                total_cores = os.cpu_count()
                initial_threads_per_process = total_cores // world_size
                torch.set_num_threads(8)

                all_embeddings = []
                all_ids = []
                for cand_pool_name in cand_pool_name_list:
                    cand_pool_name = cand_pool_name.lower()
                    cand_pool_name = f"mbeir_{cand_pool_name}_{split_name}"
                    embed_data_name = f"{cand_pool_name}_embed.npy"
                    id_data_name = f"{cand_pool_name}_ids.npy"
                    embed_path = os.path.join(
                        uniir_dir,
                        embed_dir_name,
                        expt_dir_name,
                        split_name,
                        embed_data_name,
                    )
                    id_path = os.path.join(
                        uniir_dir,
                        embed_dir_name,
                        expt_dir_name,
                        split_name,
                        id_data_name,
                    )
                    all_embeddings.append(np.load(embed_path))
                    all_ids.append(np.load(id_path))
                    print(f"Embedder Log: Concatenating embeddings from {embed_path} and ids from {id_path}.")

                all_embeddings = np.concatenate(all_embeddings, axis=0)
                all_ids = np.concatenate(all_ids, axis=0)
                assert len(all_embeddings) == len(all_ids), "Mismatch between embeddings and IDs length."
                print(f"Embedder Log: all_embeddings length: {len(all_embeddings)} and all_ids length: {len(all_ids)}.")

                # Save the embeddings to .npy
                embed_data_name = f"mbeir_union_{split_name}_embed.npy"
                embed_path = os.path.join(
                    uniir_dir,
                    embed_dir_name,
                    expt_dir_name,
                    split_name,
                    embed_data_name,
                )
                os.makedirs(os.path.dirname(embed_path), exist_ok=True)
                np.save(embed_path, all_embeddings)
                print(f"Embedder Log: Saved embeddings to {embed_path}.")

                # Save the IDs to .npy
                id_data_name = f"mbeir_union_{split_name}_ids.npy"
                id_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, all_ids)
                print(f"Embedder Log: Saved ids to {id_path}.")

                # Delete the embeddings and IDs to free up memory
                del all_embeddings
                del all_ids

                # Explicitly call the garbage collector
                gc.collect()

                # Reset number of threads to initial value after conversion
                torch.set_num_threads(8)

            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.


def main(config, lora_checkpoint=None):
    # Ensure qwen_vl_utils is importable (needed by Qwen3-VL-Embedding)
    _ensure_qwen_vl_utils_available()

    # Set up seed for reproducibility
    seed = config.seed + dist_utils.get_rank()
    set_seed(seed)

    # Initialize and load model
    model = build_model_from_config(config)
    
    # 动态挂载防止 OOM 的约束参数
    model.max_token_length = getattr(config, "max_token_length", 1500)
    model.max_visual_pixels = getattr(config, "max_visual_pixels", 501760)

    # NOTE: build_model_from_config (utils.py) already calls setup_thought_tokens
    # when mbeir_num_thought_tokens > 0. We must NOT call it again here, or the
    # ReasoningTokenEmbeddingWrapper will be double-wrapped, breaking embedding lookup.
    num_thought_tokens = getattr(config.model, "mbeir_num_thought_tokens", 0)
    if dist_utils.is_main_process() and num_thought_tokens > 0:
        print(f"Thought tokens ({num_thought_tokens}) already set up by build_model_from_config.")

    if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        if dist_utils.is_main_process():
            print(f"Loading LoRA checkpoint from: {lora_checkpoint}")
            # Force overwrite: embeddings on disk (if any) are stale for this checkpoint.
            if hasattr(config, "embed_config"):
                config.embed_config.overwrite = True

        model_name = getattr(config.model, "name", "")
        if "qwen" in model_name.lower() or model_name == "Qwen3VLThoughtWrapper":
            from peft import LoraConfig, get_peft_model
            
            # 记录加载前的特殊 Token 权重 Norm
            with torch.no_grad():
                if hasattr(model.model.get_input_embeddings(), "special_embedding"):
                    pre_load_norm = model.model.get_input_embeddings().special_embedding.weight.norm().item()
                else:
                    pre_load_norm = 0.0

            if dist_utils.is_main_process():
                print(f"Applying PEFT LoRA manually to language_model...")

            # ================== [核心修复 2: 与 train.py 对齐的 LoRA 配置] ==================
            lora_config = LoraConfig(
                r=getattr(config.model, "lora_r", 16),
                lora_alpha=getattr(config.model, "lora_alpha", 32),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="FEATURE_EXTRACTION",
                # 注意：因为你在 train.py 使用了 ReasoningTokenEmbeddingWrapper，
                # 所以我们不需要 modules_to_save=["embed_tokens"]，包装器自己会接管特殊 Token 的权重。
            )
            # 必须挂载在 language_model 上，和 train.py 保持绝对一致！
            model.model.language_model = get_peft_model(model.model.language_model, lora_config)
            # ==============================================================================

            # ================== [核心修复 3: 读取正确的 State Dict Key] ==================
            state_dict = torch.load(lora_checkpoint, map_location="cpu", weights_only=False)
            if "trainable_state_dict" in state_dict:
                # 你的 train.py 存的是这个 key！
                state_dict = state_dict["trainable_state_dict"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                
            # 直接加载进外层 wrapper model，因为 train_one_epoch 保存时也是从这层提取的
            msg = model.load_state_dict(state_dict, strict=False)
            # ==============================================================================
            
            # 检验是否成功
            with torch.no_grad():
                if hasattr(model.model.get_input_embeddings(), "special_embedding"):
                    post_load_norm = model.model.get_input_embeddings().special_embedding.weight.norm().item()
                else:
                    post_load_norm = 0.0
            
            if dist_utils.is_main_process():
                print(f"[DEBUG] Load Message: {msg}")
                print(f"[DEBUG] Special Token Embed Norm Before: {pre_load_norm:.4f}")
                print(f"[DEBUG] Special Token Embed Norm After:  {post_load_norm:.4f}")
                if abs(pre_load_norm - post_load_norm) < 1e-6 and num_thought_tokens > 0:
                    print("🚨 警告：<final> Token 权重没变！你的 Token 权重并没有被加载！")
                else:
                    print("✅ 完美成功：LoRA 和 Thought Tokens 已全部生效！")

        elif model_name == "JinaEmbeddingsV4opModel":
            from models.jina_v4op.train import load_lora_checkpoint
            model, _ = load_lora_checkpoint(model, lora_checkpoint)
        else:
            from models.jina_v4.train import load_lora_checkpoint, freeze_base_model_keep_lora
            model = freeze_base_model_keep_lora(model)
            model, _ = load_lora_checkpoint(model, lora_checkpoint)
        
        if dist_utils.is_main_process():
            print("Checkpoint loading routine finished!")
    
    model.eval()

    if not callable(getattr(model, "encode_mbeir_batch")):
        raise AttributeError("The provided model does not have a callable 'encode_mbeir_batch' method.")
    if not callable(getattr(model, "get_img_preprocess_fn")):
        raise AttributeError("The provided model does not have an 'img_preprocess_fn' attribute.")
    if not callable(getattr(model, "get_tokenizer")):
        raise AttributeError("The provided model does not have a 'tokenizer' attribute.")
    
    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()

    # Inference 时不能使用 DDP
    model = model.to(config.dist_config.gpu_id)
    print(f"Model is set up on GPU {config.dist_config.gpu_id}.")

    # Generate embeddings
    generate_embeds_for_config(
        model=model,
        img_preprocess_fn=img_preprocess_fn,
        tokenizer=tokenizer,
        config=config,
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Embeddings for MBEIR")
    parser.add_argument("--uniir_dir", type=str, default="/data/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="/data/UniIR/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument("--reason_steps", type=int, default=0, 
                        help="(Deprecated) Number of implicit reasoning steps in embedding space (0 = disabled)")
    parser.add_argument("--num_thought_tokens", type=int, default=None,
                        help="Number of thought tokens for reasoning (overrides config if set)")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to fine-tuned LoRA checkpoint to load (optional)")
    parser.add_argument("--max_token_length", type=int, default=1500)
    parser.add_argument("--max_visual_pixels", type=int, default=501760)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embeddings even if .npy files already exist",
    )
    return parser.parse_args()


if __name__ == "__main__":

    ### 解决 transformers tokenizers 并行警告问题
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ###

    args = parse_arguments()
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir
    config.max_token_length = args.max_token_length
    config.max_visual_pixels = args.max_visual_pixels
    if not hasattr(config, "embed_config"):
        config.embed_config = OmegaConf.create({})
    config.embed_config.overwrite = bool(args.overwrite)

    # Set reason_steps in model config (will be read by build_model_from_config)
    if not hasattr(config, 'model'):
        config.model = OmegaConf.create({})
    config.model.mbeir_reason_steps = args.reason_steps
    
    # Set num_thought_tokens if provided via command line (overrides config)
    if args.num_thought_tokens is not None:
        config.model.mbeir_num_thought_tokens = args.num_thought_tokens
    
    # Determine the steps value for path suffix
    # Priority: num_thought_tokens > reason_steps
    steps_for_path = args.num_thought_tokens if args.num_thought_tokens is not None else args.reason_steps
    
    # Update path_suffix to include steps for unique output directories
    if steps_for_path > 0:
        original_path_suffix = config.experiment.path_suffix
        config.experiment.path_suffix = f"{original_path_suffix}reason_steps_{steps_for_path}/"

    # Initialize distributed mode
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    dist_utils.init_distributed_mode(args)

    if not getattr(args, "distributed", False):
        # dist_utils.init_distributed_mode returns early and does NOT set args.gpu
        args.gpu = getattr(config.dist_config, "gpu_id", 0)

    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = getattr(args, "distributed", False)

    if dist_utils.is_main_process():
        if args.num_thought_tokens is not None:
            print(f"Running with num_thought_tokens={args.num_thought_tokens}")
        else:
            print(f"Running with reason_steps={args.reason_steps}")
        if args.lora_checkpoint:
            print(f"LoRA checkpoint: {args.lora_checkpoint}")
        print(f"Output path_suffix: {config.experiment.path_suffix}")
        print(OmegaConf.to_yaml(config, sort_keys=False))

    main(config, lora_checkpoint=args.lora_checkpoint)

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()