"""
Per-step thought token embedder for truncated reasoning analysis.

This embedder is based on mbeir_embedder.py but adds support for extracting
embeddings at a specific thought token position. This enables evaluation of
retrieval performance at each intermediate reasoning step.

Supports two modes:
  1. Single-step mode (--thought_step N): encode queries using step N only
  2. All-steps mode (--all_steps): single forward pass, save all K steps at once

Usage (single step):
    python -m torch.distributed.run --nproc_per_node=2 \
        mbeir_embedder_per_step.py \
        --config_path embed.yaml \
        --uniir_dir /data/LR1/runs/eval_truncation/step_3 \
        --num_thought_tokens 5 --thought_step 3 \
        --lora_checkpoint /path/to/checkpoint.pth

Usage (all steps at once):
    python -m torch.distributed.run --nproc_per_node=2 \
        mbeir_embedder_per_step.py \
        --config_path embed.yaml \
        --uniir_dir /data/LR1/runs/eval_truncation \
        --num_thought_tokens 5 --all_steps \
        --lora_checkpoint /path/to/checkpoint.pth
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
from torch.amp import autocast
import transformers

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
def generate_embeds_and_ids_with_thought_step(
    model, data_loader, device, thought_step=0, use_fp16=True
):
    """
    Generate embeddings using a specific thought step for queries.

    When thought_step > 0, query batches use the thought_step-th thought token's
    embedding. Candidate batches always use standard encoding.
    """
    embedding_tensors = []
    id_list = []

    total_cores = os.cpu_count()
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # 强行限制每个 GPU 进程最多只使用 2 个 CPU 线程进行张量计算
        initial_threads_per_process = 2 
        torch.set_num_threads(initial_threads_per_process)
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank} (step={thought_step})")
    for batch in data_loader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(device)

        with autocast(device_type="cuda", enabled=use_fp16):
            if thought_step > 0:
                embeddings_batched, ids_list_batched = model.encode_mbeir_batch_at_step(
                    batch, thought_step
                )
            else:
                embeddings_batched, ids_list_batched = model(
                    encode_mbeir_batch=True, **batch
                )

        embedding_tensors.append(embeddings_batched.half())
        id_list.extend(ids_list_batched)

    embedding_tensor = torch.cat(embedding_tensors, dim=0)
    embedding_list = None

    if dist.is_initialized():
        size_tensor = torch.tensor([embedding_tensor.size(0)], dtype=torch.long, device=device)

        if dist.get_rank() == 0:
            gathered_embeddings = [torch.empty_like(embedding_tensor) for _ in range(dist.get_world_size())]
            id_list_gathered = [list() for _ in range(dist.get_world_size())]
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
        else:
            gathered_embeddings = None
            id_list_gathered = None
            sizes = None

        dist.barrier()
        dist.gather(embedding_tensor, gather_list=gathered_embeddings, dst=0)
        dist.gather_object(id_list, object_gather_list=id_list_gathered, dst=0)
        dist.gather(size_tensor, gather_list=sizes, dst=0)
        dist.barrier()

        if dist.get_rank() == 0:
            print(f"Embedder Log: Gathered embeddings on rank 0 (thought_step={thought_step})...")
            torch.set_num_threads(8)
            gathered_embeddings[-1] = gathered_embeddings[-1][: sizes[-1][0]]
            embedding_list = torch.cat(gathered_embeddings, dim=0)
            id_list = [id for sublist in id_list_gathered for id in sublist]
            assert len(id_list) == embedding_list.size(0)
            assert len(id_list) == len(set(id_list)), "Hashed IDs should be unique"
            print(f"Embedder Log: Finished processing embeddings on rank 0.")
            embedding_list = embedding_list.half().cpu().numpy()
            print(f"Embedder Log: Converted to cpu numpy array of type {embedding_list.dtype}.")
            torch.set_num_threads(initial_threads_per_process)

        dist.barrier()
    else:
        embedding_list = embedding_tensor.half().cpu().numpy()

    return embedding_list, id_list


@torch.no_grad()
def generate_embeds_and_ids_all_steps(
    model, data_loader, device, num_steps, use_fp16=True
):
    """
    Generate embeddings for ALL thought steps in a single forward pass.

    Uses model.encode_mbeir_batch_all_steps() which runs one forward pass and
    returns K embedding tensors (one per thought token position).

    Returns:
        (results_per_step, id_list) where results_per_step is a dict mapping
        step (1-indexed) to numpy array of embeddings.
    """
    embedding_tensors_per_step = [[] for _ in range(num_steps)]
    id_list = []

    total_cores = os.cpu_count()
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        initial_threads_per_process = total_cores // world_size
        torch.set_num_threads(initial_threads_per_process)
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank} (all {num_steps} steps)")

    for batch in data_loader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(device)

        with autocast(device_type="cuda", enabled=use_fp16):
            all_embeddings, ids_list_batched = model.encode_mbeir_batch_all_steps(batch)

        for step_idx in range(num_steps):
            embedding_tensors_per_step[step_idx].append(all_embeddings[step_idx].half())
        id_list.extend(ids_list_batched)

    # Concatenate each step's batch embeddings
    cat_tensors = [torch.cat(step_list, dim=0) for step_list in embedding_tensors_per_step]
    del embedding_tensors_per_step

    results_per_step = {}

    if dist.is_initialized():
        # All steps share the same local sample count
        size_tensor = torch.tensor([cat_tensors[0].size(0)], dtype=torch.long, device=device)

        if dist.get_rank() == 0:
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
            id_list_gathered = [list() for _ in range(dist.get_world_size())]
        else:
            sizes = None
            id_list_gathered = None

        dist.barrier()
        dist.gather(size_tensor, gather_list=sizes, dst=0)
        dist.gather_object(id_list, object_gather_list=id_list_gathered, dst=0)

        # Gather each step's embeddings
        for step_idx in range(num_steps):
            if dist.get_rank() == 0:
                gathered = [torch.empty_like(cat_tensors[step_idx]) for _ in range(dist.get_world_size())]
            else:
                gathered = None

            dist.gather(cat_tensors[step_idx], gather_list=gathered, dst=0)

            if dist.get_rank() == 0:
                gathered[-1] = gathered[-1][:sizes[-1][0]]
                result = torch.cat(gathered, dim=0).half().cpu().numpy()
                results_per_step[step_idx + 1] = result
                del gathered

        dist.barrier()

        if dist.get_rank() == 0:
            torch.set_num_threads(8)
            id_list = [id for sublist in id_list_gathered for id in sublist]
            assert len(id_list) == results_per_step[1].shape[0]
            assert len(id_list) == len(set(id_list)), "Hashed IDs should be unique"
            print(f"Embedder Log: Gathered embeddings for all {num_steps} steps on rank 0.")
            print(f"Embedder Log: {len(id_list)} samples, {results_per_step[1].dtype} dtype.")
            torch.set_num_threads(initial_threads_per_process)
    else:
        for step_idx in range(num_steps):
            results_per_step[step_idx + 1] = cat_tensors[step_idx].half().cpu().numpy()

    del cat_tensors
    return results_per_step, id_list


@torch.no_grad()
def generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config, thought_step=0):
    """
    Generate embeddings for queries and candidates (single-step mode).

    For query datasets, uses thought_step to select which thought token embedding to use.
    For candidate pools, always uses standard encoding (thought_step=0).
    """
    uniir_dir = config.uniir_dir
    mbeir_data_dir = config.mbeir_data_dir
    embed_config = config.embed_config
    embed_dir_name = embed_config.embed_dir_name
    expt_dir_name = config.experiment.path_suffix

    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))

    splits = []
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        split_dir_name = getattr(data_config, f"{split_name}_dir_name")
        embed_dataset_config = getattr(embed_config, f"{split_name}_datasets_config", None)
        if embed_dataset_config and embed_dataset_config.enable_embed:
            dataset_name_list = getattr(embed_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(embed_dataset_config, "correspond_cand_pools_name", None)
            splits.append((split_name, split_dir_name, dataset_name_list, cand_pool_name_list))
            assert len(dataset_name_list) == len(cand_pool_name_list)

    embed_cand_pool_config = embed_config.cand_pools_config
    if embed_cand_pool_config and embed_cand_pool_config.enable_embed:
        split_name = "cand_pool"
        split_dir_name = data_config.cand_pool_dir_name
        cand_pool_name_list = embed_cand_pool_config.cand_pools_name_to_embed
        splits.append((split_name, split_dir_name, [None] * len(cand_pool_name_list), cand_pool_name_list))

    if dist_utils.is_main_process():
        print("-" * 30)
        for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
            if split_name == "cand_pool":
                print(f"Split: {split_name}, Candidate pools: {cand_pool_name_list}")
            else:
                print(f"Split: {split_name}, Datasets: {dataset_name_list} (thought_step={thought_step})")
            print("-" * 30)

    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        current_thought_step = thought_step if split_name != "cand_pool" else 0

        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            if split_name == "cand_pool":
                mid_name = cand_pool_name.lower()
            else:
                mid_name = dataset_name.lower() if dataset_name else None

            if mid_name:
                embed_data_name = f"mbeir_{mid_name}_{split_name}_embed.npy"
                id_data_name = f"mbeir_{mid_name}_{split_name}_ids.npy"
                embed_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, embed_data_name)
                id_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, id_data_name)

                if os.path.exists(embed_path) and os.path.exists(id_path):
                    if dist_utils.is_main_process():
                        print(f"\n[SKIP] Embeddings already exist: {embed_path}")
                    if dist.is_initialized():
                        dist.barrier()
                    continue

            if split_name == "cand_pool":
                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_{split_name}.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = dist_utils.is_main_process()
                if print_config:
                    print(f"\nEmbedder Log: Generating embeddings for {cand_pool_data_path}...")

                dataset = MBEIRCandidatePoolDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    cand_pool_data_path=cand_pool_data_path,
                    img_preprocess_fn=img_preprocess_fn,
                    print_config=print_config,
                )
                collator = MBEIRCandidatePoolCollator(tokenizer=tokenizer, image_size=image_size)
            else:
                dataset_name = dataset_name.lower()
                query_data_name = f"mbeir_{dataset_name}_{split_name}.jsonl"
                query_data_path = os.path.join(split_dir, query_data_name)

                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = dist_utils.is_main_process()
                if print_config:
                    print(f"\nEmbedder Log: Generating embeddings for {query_data_path} "
                          f"(thought_step={current_thought_step})...")

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
                collator = MBEIRMainCollator(tokenizer=tokenizer, image_size=image_size, mode=mode)

            batch_size = config.dataloader_config.batch_size
            num_workers = config.dataloader_config.num_workers

            num_tasks = dist_utils.get_world_size()
            global_rank = dist_utils.get_rank()
            sampler = ContiguousDistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank)
            data_loader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers,
                pin_memory=True, sampler=sampler, shuffle=False,
                collate_fn=collator, drop_last=False,
            )
            if dist.is_initialized():
                dist.barrier()
            if dist_utils.is_main_process():
                print(f"Embedder Log: Data loader set up. FP16={config.embed_config.use_fp16}")

            embedding_list, id_list = generate_embeds_and_ids_with_thought_step(
                model, data_loader,
                device=config.dist_config.gpu_id,
                thought_step=current_thought_step,
                use_fp16=config.embed_config.use_fp16,
            )

            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Embedder Log: Embedding list length: {len(embedding_list)}")
                print(f"Embedder Log: ID list length: {len(id_list)}")

                mid_name = cand_pool_name if split_name == "cand_pool" else dataset_name
                embed_data_name = f"mbeir_{mid_name}_{split_name}_embed.npy"
                embed_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, embed_data_name)
                os.makedirs(os.path.dirname(embed_path), exist_ok=True)
                np.save(embed_path, embedding_list)
                print(f"Embedder Log: Saved embeddings to {embed_path}.")

                id_data_name = f"mbeir_{mid_name}_{split_name}_ids.npy"
                id_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, id_list)
                print(f"Embedder Log: Saved ids to {id_path}.")

            if dist.is_initialized():
                dist.barrier()

            del embedding_list, id_list, data_loader, dataset, collator, sampler
            gc.collect()
            torch.cuda.empty_cache()

        # Union pool (same as standard embedder)
        if split_name == "cand_pool" and embed_cand_pool_config.embed_union_pool:
            union_embed_path = os.path.join(
                uniir_dir, embed_dir_name, expt_dir_name, split_name, f"mbeir_union_{split_name}_embed.npy"
            )
            union_id_path = os.path.join(
                uniir_dir, embed_dir_name, expt_dir_name, split_name, f"mbeir_union_{split_name}_ids.npy"
            )

            if os.path.exists(union_embed_path) and os.path.exists(union_id_path):
                if dist_utils.is_main_process():
                    print(f"\n[SKIP] Union pool embeddings already exist: {union_embed_path}")
                if dist.is_initialized():
                    dist.barrier()
                continue

            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\nEmbedder Log: Generating union pool embeddings...")
                world_size = dist.get_world_size()
                total_cores = os.cpu_count()
                initial_threads_per_process = total_cores // world_size
                torch.set_num_threads(8)

                all_embeddings = []
                all_ids = []
                for cand_pool_name in cand_pool_name_list:
                    cand_pool_name = cand_pool_name.lower()
                    embed_path = os.path.join(
                        uniir_dir, embed_dir_name, expt_dir_name, split_name,
                        f"mbeir_{cand_pool_name}_{split_name}_embed.npy"
                    )
                    id_path = os.path.join(
                        uniir_dir, embed_dir_name, expt_dir_name, split_name,
                        f"mbeir_{cand_pool_name}_{split_name}_ids.npy"
                    )
                    all_embeddings.append(np.load(embed_path))
                    all_ids.append(np.load(id_path))

                all_embeddings = np.concatenate(all_embeddings, axis=0)
                all_ids = np.concatenate(all_ids, axis=0)
                assert len(all_embeddings) == len(all_ids)

                np.save(union_embed_path, all_embeddings)
                np.save(union_id_path, all_ids)
                print(f"Embedder Log: Saved union pool to {union_embed_path}.")

                del all_embeddings, all_ids
                gc.collect()
                torch.set_num_threads(initial_threads_per_process)

            if dist.is_initialized():
                dist.barrier()


@torch.no_grad()
def generate_embeds_for_config_all_steps(model, img_preprocess_fn, tokenizer, config, num_steps):
    """
    Generate embeddings for ALL thought steps in a single encoding pass.

    For each query dataset, runs one forward pass per batch and saves K sets of
    embeddings to step_1/, step_2/, ..., step_K/ subdirectories under uniir_dir.

    Candidate pools are skipped (expected to be reused via symlink).
    """
    uniir_dir = config.uniir_dir  # base work dir, e.g. .../eval_thought_truncation/MTtrunO5
    mbeir_data_dir = config.mbeir_data_dir
    embed_config = config.embed_config
    embed_dir_name = embed_config.embed_dir_name
    expt_dir_name = config.experiment.path_suffix

    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))

    splits = []
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        split_dir_name = getattr(data_config, f"{split_name}_dir_name")
        embed_dataset_config = getattr(embed_config, f"{split_name}_datasets_config", None)
        if embed_dataset_config and embed_dataset_config.enable_embed:
            dataset_name_list = getattr(embed_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(embed_dataset_config, "correspond_cand_pools_name", None)
            splits.append((split_name, split_dir_name, dataset_name_list, cand_pool_name_list))
            assert len(dataset_name_list) == len(cand_pool_name_list)

    if dist_utils.is_main_process():
        print("-" * 30)
        for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
            print(f"Split: {split_name}, Datasets: {dataset_name_list} (all {num_steps} steps)")
            print("-" * 30)

    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            dataset_name = dataset_name.lower()
            mid_name = dataset_name

            # Check if ALL steps already exist
            all_exist = True
            for step in range(1, num_steps + 1):
                step_uniir_dir = os.path.join(uniir_dir, f"step_{step}")
                embed_data_name = f"mbeir_{mid_name}_{split_name}_embed.npy"
                id_data_name = f"mbeir_{mid_name}_{split_name}_ids.npy"
                embed_path = os.path.join(step_uniir_dir, embed_dir_name, expt_dir_name, split_name, embed_data_name)
                id_path = os.path.join(step_uniir_dir, embed_dir_name, expt_dir_name, split_name, id_data_name)
                if not (os.path.exists(embed_path) and os.path.exists(id_path)):
                    all_exist = False
                    break

            if all_exist:
                if dist_utils.is_main_process():
                    print(f"\n[SKIP] Embeddings for all {num_steps} steps already exist for {mid_name}")
                if dist.is_initialized():
                    dist.barrier()
                continue

            # Build dataset and dataloader
            query_data_name = f"mbeir_{dataset_name}_{split_name}.jsonl"
            query_data_path = os.path.join(split_dir, query_data_name)

            cand_pool_name = cand_pool_name.lower()
            cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
            cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

            print_config = dist_utils.is_main_process()
            if print_config:
                print(f"\nEmbedder Log: Generating all-steps embeddings for {query_data_path}...")

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
            collator = MBEIRMainCollator(tokenizer=tokenizer, image_size=image_size, mode=mode)

            batch_size = config.dataloader_config.batch_size
            num_workers = config.dataloader_config.num_workers

            num_tasks = dist_utils.get_world_size()
            global_rank = dist_utils.get_rank()
            sampler = ContiguousDistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank)
            data_loader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers,
                pin_memory=True, sampler=sampler, shuffle=False,
                collate_fn=collator, drop_last=False,
            )
            if dist.is_initialized():
                dist.barrier()
            if dist_utils.is_main_process():
                print(f"Embedder Log: Data loader set up. FP16={config.embed_config.use_fp16}")

            results_per_step, id_list = generate_embeds_and_ids_all_steps(
                model, data_loader,
                device=config.dist_config.gpu_id,
                num_steps=num_steps,
                use_fp16=config.embed_config.use_fp16,
            )

            # Save results for each step to separate directories
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Embedder Log: ID list length: {len(id_list)}")
                for step, embedding_list in results_per_step.items():
                    print(f"Embedder Log: Step {step} embedding shape: {embedding_list.shape}")

                    step_uniir_dir = os.path.join(uniir_dir, f"step_{step}")

                    embed_data_name = f"mbeir_{dataset_name}_{split_name}_embed.npy"
                    embed_path = os.path.join(
                        step_uniir_dir, embed_dir_name, expt_dir_name, split_name, embed_data_name
                    )
                    os.makedirs(os.path.dirname(embed_path), exist_ok=True)
                    np.save(embed_path, embedding_list)

                    id_data_name = f"mbeir_{dataset_name}_{split_name}_ids.npy"
                    id_path = os.path.join(
                        step_uniir_dir, embed_dir_name, expt_dir_name, split_name, id_data_name
                    )
                    os.makedirs(os.path.dirname(id_path), exist_ok=True)
                    np.save(id_path, id_list)

                print(f"Embedder Log: Saved embeddings for all {num_steps} steps.")

            if dist.is_initialized():
                dist.barrier()

            del results_per_step, id_list, data_loader, dataset, collator, sampler
            gc.collect()
            torch.cuda.empty_cache()


def main(config, lora_checkpoint=None, thought_step=0, all_steps=False, num_steps=0):
    seed = config.seed + dist_utils.get_rank()
    set_seed(seed)

    model = build_model_from_config(config)

    if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        if dist_utils.is_main_process():
            print(f"Loading LoRA checkpoint from: {lora_checkpoint}")

        model_name = config.model.name
        if model_name == "JinaEmbeddingsV4oModel":
            from models.jina_v4o.train import load_lora_checkpoint
            model, _ = load_lora_checkpoint(model, lora_checkpoint)
        else:
            from models.jina_v4.train import load_lora_checkpoint, freeze_base_model_keep_lora
            model = freeze_base_model_keep_lora(model)
            model, _ = load_lora_checkpoint(model, lora_checkpoint)

        if dist_utils.is_main_process():
            print("Checkpoint loaded successfully!")

    model.eval()
    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()

    model = model.to(config.dist_config.gpu_id)
    print(f"Model is set up on GPU {config.dist_config.gpu_id}.")

    if all_steps:
        if dist_utils.is_main_process():
            print(f"=== All-steps mode: encoding all {num_steps} steps in a single pass ===")
        generate_embeds_for_config_all_steps(
            model=model,
            img_preprocess_fn=img_preprocess_fn,
            tokenizer=tokenizer,
            config=config,
            num_steps=num_steps,
        )
    else:
        generate_embeds_for_config(
            model=model,
            img_preprocess_fn=img_preprocess_fn,
            tokenizer=tokenizer,
            config=config,
            thought_step=thought_step,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Per-step Thought Token Embedder for Truncated Reasoning Analysis"
    )
    parser.add_argument("--uniir_dir", type=str, default="/data/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="/data/UniIR/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml")
    parser.add_argument("--num_thought_tokens", type=int, default=5,
                        help="Total number of thought tokens (must match training)")
    parser.add_argument("--thought_step", type=int, default=0,
                        help="Which thought step to use for query embedding (1-indexed, 0=standard)")
    parser.add_argument("--all_steps", action="store_true",
                        help="Encode all steps in a single forward pass and save separately")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to fine-tuned LoRA checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_arguments()
    config = OmegaConf.load(args.config_path)

    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    if not hasattr(config, 'model'):
        config.model = OmegaConf.create({})

    # Always set up all thought tokens (matching training checkpoint)
    config.model.mbeir_num_thought_tokens = args.num_thought_tokens

    # Update path_suffix to include thought tokens for unique output directories
    if args.num_thought_tokens > 0:
        original_path_suffix = config.experiment.path_suffix
        config.experiment.path_suffix = f"{original_path_suffix}reason_steps_{args.num_thought_tokens}/"

    # Initialize distributed mode
    args.dist_url = config.dist_config.dist_url
    dist_utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    if dist_utils.is_main_process():
        print(f"=== Per-Step Thought Token Embedder ===")
        if args.all_steps:
            print(f"Mode: ALL STEPS (num_thought_tokens={args.num_thought_tokens})")
        else:
            print(f"Mode: SINGLE STEP (thought_step={args.thought_step})")
        print(f"num_thought_tokens={args.num_thought_tokens}")
        if args.lora_checkpoint:
            print(f"LoRA checkpoint: {args.lora_checkpoint}")
        print(f"Output path_suffix: {config.experiment.path_suffix}")
        print(OmegaConf.to_yaml(config, sort_keys=False))

    main(
        config,
        lora_checkpoint=args.lora_checkpoint,
        thought_step=args.thought_step,
        all_steps=args.all_steps,
        num_steps=args.num_thought_tokens,
    )

    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
