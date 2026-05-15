"""
Case Selector: Find query cases across M-BEIR datasets for analysis.
=================================================================
Supports selecting:
1. MT5 recalls a relevant document while MT0 does not.
2. Both MT5 and MT0 fail to recall any relevant document.

Usage examples:
    python case_selector.py --dataset webqa_task2 --selection_mode mt5_recall_mt0_miss
    python case_selector.py --all_datasets --selection_mode mt5_recall_mt0_miss --selection_mode both_miss
    python case_selector.py --all_datasets --selection_mode mt5_recall_mt0_miss
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Constants: Default paths
# ============================================================================
MBEIR_DATA_DIR = "/data/M-BEIR/sub_MBEIR"
QUERY_TEST_DIR = os.path.join(MBEIR_DATA_DIR, "query/test")
QRELS_TEST_DIR = os.path.join(MBEIR_DATA_DIR, "qrels/test")

ALL_QUERY_DATASETS = [
    "fashion200k_task0",
    "fashion200k_task3",
    "infoseek_task6",
    "mscoco_task0",
    "mscoco_task3",
    "oven_task6",
    "visualnews_task0",
    "visualnews_task3",
    "webqa_task2",
]

BASELINE_RUN_DIR = (
    "/data/LR1/runs/eval_finetunedtt/MT0eop3/retrieval_results/"
    "JinaV4/Large/Instruct/InBatch/run_files"
)
# MT5_RUN_DIR = (
#     "/data/LR1/runs/eval_finetuned/MT5epo3/retrieval_results/"
#     "JinaV4/Large/Instruct/InBatch/reason_steps_5/run_files"
# )
MT5_RUN_DIR = (
    "/data/LR1/runs/eval_thought_truncation/MTtrunO5/step_5/retrieval_results/"
    "JinaV4/Large/Instruct/InBatch/reason_steps_5/run_files"
)

OUTPUT_DIR = "/data/LR1/src/models/jina_v4o/analysis/output"
SELECTION_MODES = ["mt5_recall_mt0_miss", "both_miss"]


# ============================================================================
# Parsing Functions
# ============================================================================

def load_run_file(filepath: str) -> Dict[str, List[dict]]:
    """
    Load a TREC-format run file.
    
    Format per line: qid Q0 did rank score run_name task_id
    
    Returns:
        Dict[qid -> List[{did, rank, score}]] sorted by rank ascending.
    """
    results = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, did, rank, score = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4])
            results[qid].append({"did": did, "rank": rank, "score": score})
    
    # Sort each query's results by rank
    for qid in results:
        results[qid].sort(key=lambda x: x["rank"])
    
    return dict(results)


def load_qrels(filepath: str) -> Dict[str, List[str]]:
    """
    Load qrels file (ground-truth relevance judgments).
    
    Format per line: qid 0 did relevance task_id
    
    Returns:
        Dict[qid -> List[relevant_did]]
    """
    qrels = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, did, relevance = parts[0], parts[1], parts[2], int(parts[3])
            if relevance > 0:
                qrels[qid].append(did)
    return dict(qrels)


def load_query_data(filepath: str) -> Dict[str, dict]:
    """
    Load query test JSONL file.
    
    Returns:
        Dict[qid -> query_entry_dict]
    """
    queries = {}
    with open(filepath, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            qid = entry["qid"]
            queries[qid] = entry
    return queries


# ============================================================================
# Core Logic
# ============================================================================

def get_best_relevant_rank(
    run_results: List[dict],
    relevant_dids: List[str],
) -> Optional[int]:
    """
    Find the best (lowest) rank at which any relevant document appears.

    Args:
        run_results: Sorted list of {did, rank, score} for one query.
        relevant_dids: Set of ground-truth relevant document IDs.

    Returns:
        Best rank (1-indexed) if found, else None (not retrieved).
    """
    relevant_set = set(relevant_dids)
    for item in run_results:
        if item["did"] in relevant_set:
            return item["rank"]
    return None  # Not found in top-K


def get_k_value(dataset: str) -> str:
    return "k50" if "fashion200k" in dataset else "k10"


def build_run_filename(dataset: str, pool: str) -> str:
    return f"mbeir_{dataset}_{pool}_pool_test_{get_k_value(dataset)}_run.txt"


def build_case_entry(
    dataset: str,
    qid: str,
    query_info: dict,
    relevant_dids: List[str],
    baseline_runs: Dict[str, List[dict]],
    mt5_runs: Dict[str, List[dict]],
    baseline_rank: Optional[int],
    mt5_rank: Optional[int],
    selection_mode: str,
    max_rank_for_fail: int,
) -> dict:
    baseline_effective = baseline_rank if baseline_rank is not None else max_rank_for_fail
    mt5_effective = mt5_rank if mt5_rank is not None else max_rank_for_fail
    delta_rank = baseline_effective - mt5_effective

    case = {
        "dataset": dataset,
        "selection_mode": selection_mode,
        "qid": qid,
        "query_txt": query_info.get("query_txt", ""),
        "query_img_path": query_info.get("query_img_path", None),
        "query_modality": query_info.get("query_modality", ""),
        "task_id": query_info.get("task_id", None),
        "mt0_rank": baseline_rank,
        "mt5_rank": mt5_rank,
        "delta_rank": delta_rank,
        "mt0_effective_rank": baseline_effective,
        "mt5_effective_rank": mt5_effective,
        "relevant_dids": relevant_dids,
        "mt0_top5": [r["did"] for r in baseline_runs.get(qid, [])[:5]],
        "mt5_top5": [r["did"] for r in mt5_runs.get(qid, [])[:5]],
        "baseline_tag": "MT0",
        "improved_tag": "MT5",
    }

    # Backward-compatible aliases for downstream code that may inspect old keys.
    case["mt1_rank"] = case["mt0_rank"]
    case["mt1_effective_rank"] = case["mt0_effective_rank"]
    case["mt1_top5"] = case["mt0_top5"]
    return case


def select_cases_by_mode(
    baseline_runs: Dict[str, List[dict]],
    mt5_runs: Dict[str, List[dict]],
    qrels: Dict[str, List[str]],
    queries: Dict[str, dict],
    dataset: str,
    selection_mode: str,
    max_rank_for_fail: int = 10000,
) -> List[dict]:
    cases = []
    common_qids = set(baseline_runs.keys()) & set(mt5_runs.keys()) & set(qrels.keys())

    for qid in common_qids:
        relevant_dids = qrels[qid]
        baseline_rank = get_best_relevant_rank(baseline_runs[qid], relevant_dids)
        mt5_rank = get_best_relevant_rank(mt5_runs[qid], relevant_dids)

        if selection_mode == "mt5_recall_mt0_miss":
            matched = baseline_rank is None and mt5_rank is not None
        elif selection_mode == "both_miss":
            matched = baseline_rank is None and mt5_rank is None
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        if not matched:
            continue

        query_info = queries.get(qid, {})
        cases.append(
            build_case_entry(
                dataset=dataset,
                qid=qid,
                query_info=query_info,
                relevant_dids=relevant_dids,
                baseline_runs=baseline_runs,
                mt5_runs=mt5_runs,
                baseline_rank=baseline_rank,
                mt5_rank=mt5_rank,
                selection_mode=selection_mode,
                max_rank_for_fail=max_rank_for_fail,
            )
        )

    if selection_mode == "mt5_recall_mt0_miss":
        cases.sort(key=lambda x: x["delta_rank"], reverse=True)
    else:
        cases.sort(key=lambda x: (x["dataset"], x["qid"]))
    return cases


def select_cases(
    mt1_runs: Dict[str, List[dict]],
    mt5_runs: Dict[str, List[dict]],
    qrels: Dict[str, List[str]],
    queries: Dict[str, dict],
    max_rank_for_fail: int = 10000,
) -> List[dict]:
    """
    Backward-compatible wrapper for the original MT1-fail / MT5-success logic.
    """
    return select_cases_by_mode(
        baseline_runs=mt1_runs,
        mt5_runs=mt5_runs,
        qrels=qrels,
        queries=queries,
        dataset="unknown",
        selection_mode="mt5_recall_mt0_miss",
        max_rank_for_fail=max_rank_for_fail,
    )


def collect_cases_for_dataset(
    dataset: str,
    pool: str,
    selection_mode: str,
    top_n: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    run_filename = build_run_filename(dataset, pool)
    baseline_run_path = os.path.join(BASELINE_RUN_DIR, run_filename)
    mt5_run_path = os.path.join(MT5_RUN_DIR, run_filename)
    qrels_path = os.path.join(QRELS_TEST_DIR, f"mbeir_{dataset}_test_qrels.txt")
    query_path = os.path.join(QUERY_TEST_DIR, f"mbeir_{dataset}_test.jsonl")

    for path, label in [
        (baseline_run_path, "MT0 run file"),
        (mt5_run_path, "MT5 run file"),
        (qrels_path, "Qrels file"),
        (query_path, "Query file"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    baseline_runs = load_run_file(baseline_run_path)
    mt5_runs = load_run_file(mt5_run_path)
    qrels = load_qrels(qrels_path)
    queries = load_query_data(query_path)

    cases = select_cases_by_mode(
        baseline_runs=baseline_runs,
        mt5_runs=mt5_runs,
        qrels=qrels,
        queries=queries,
        dataset=dataset,
        selection_mode=selection_mode,
    )

    selected = cases[:top_n] if top_n is not None else cases
    metadata = {
        "dataset": dataset,
        "pool": pool,
        "selection_mode": selection_mode,
        "baseline_tag": "MT0",
        "improved_tag": "MT5",
        "total_case_count": len(cases),
        "selected_count": len(selected),
        "baseline_run": baseline_run_path,
        "mt5_run": mt5_run_path,
    }
    return selected, metadata


def collect_cases_for_all_datasets(
    datasets: List[str],
    pool: str,
    selection_mode: str,
    top_n: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    all_cases = []
    per_dataset_counts = {}

    for dataset in datasets:
        # 【修改点 1】: 将 top_n 传入 collect_cases_for_dataset
        # 这样每个子集在收集时就会独立截取前 50 个提升最明显的案例
        cases, _ = collect_cases_for_dataset(
            dataset=dataset,
            pool=pool,
            selection_mode=selection_mode,
            top_n=top_n, 
        )
        per_dataset_counts[dataset] = len(cases)
        all_cases.extend(cases)

    # 保留全局排序，这样在最终的 JSON 文件中，所有数据会按提升幅度从大到小排列
    if selection_mode == "mt5_recall_mt0_miss":
        all_cases.sort(key=lambda x: x["delta_rank"], reverse=True)
    else:
        all_cases.sort(key=lambda x: (x["dataset"], x["qid"]))

    # 【修改点 2】: 移除这里的全局截断，因为已经按子集截断过了
    selected = all_cases 
    metadata = {
        "datasets": datasets,
        "pool": pool,
        "selection_mode": selection_mode,
        "baseline_tag": "MT0",
        "improved_tag": "MT5",
        "total_case_count": sum(per_dataset_counts.values()), # 更新总数计算
        "selected_count": len(selected),
        "per_dataset_counts": per_dataset_counts,
        "baseline_run_dir": BASELINE_RUN_DIR,
        "mt5_run_dir": MT5_RUN_DIR,
    }
    return selected, metadata


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Select cases for attention analysis")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Single dataset to analyze (ignored if --all_datasets is set)"
    )
    parser.add_argument(
        "--all_datasets", action="store_true",
        help="Run on all M-BEIR query datasets"
    )
    parser.add_argument(
        "--pool", type=str, default="union",
        choices=["union", "single"],
        help="Candidate pool type"
    )
    parser.add_argument(
        "--selection_mode", type=str, default="mt5_recall_mt0_miss",
        choices=SELECTION_MODES,
        help="Selection mode: mt5_recall_mt0_miss or both_miss"
    )
    parser.add_argument("--top_n", type=int, default=None, help="Limit number of cases (None = all)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all_datasets:
        datasets = ALL_QUERY_DATASETS
        print(f"Selection mode: {args.selection_mode}")
        print(f"Pool: {args.pool}")
        print(f"Datasets: {len(datasets)}")
        print("-" * 60)

        selected, metadata = collect_cases_for_all_datasets(
            datasets=datasets,
            pool=args.pool,
            selection_mode=args.selection_mode,
            top_n=args.top_n,
        )

        print(f"Per-dataset counts:")
        for ds, cnt in metadata["per_dataset_counts"].items():
            print(f"  {ds}: {cnt}")
        print("-" * 60)
        print(f"Total cases: {metadata['total_case_count']}")
        print(f"Selected: {metadata['selected_count']}")

        output_path = os.path.join(
            args.output_dir,
            f"selected_cases_all_{args.pool}_{args.selection_mode}.json"
        )
    else:
        dataset = args.dataset or "webqa_task2"
        print(f"Selection mode: {args.selection_mode}")
        print(f"Dataset: {dataset}")
        print(f"Pool: {args.pool}")
        print("-" * 60)

        selected, metadata = collect_cases_for_dataset(
            dataset=dataset,
            pool=args.pool,
            selection_mode=args.selection_mode,
            top_n=args.top_n,
        )

        print(f"Total cases: {metadata['total_case_count']}")
        print(f"Selected: {metadata['selected_count']}")

        output_path = os.path.join(
            args.output_dir,
            f"selected_cases_{dataset}_{args.pool}_{args.selection_mode}.json"
        )

    # Print sample cases
    print("-" * 60)
    for i, case in enumerate(selected[:5]):
        mt0_display = case["mt0_rank"] if case["mt0_rank"] is not None else "NOT_FOUND"
        mt5_display = case["mt5_rank"] if case["mt5_rank"] is not None else "NOT_FOUND"
        print(
            f"  [{i+1}] dataset={case['dataset']}, qid={case['qid']}, "
            f"MT0_rank={mt0_display}, MT5_rank={mt5_display}, "
            f"modality={case['query_modality']}"
        )
        if case["query_txt"]:
            print(f"       Query: {case['query_txt'][:60]}...")
    if len(selected) > 5:
        print(f"  ... and {len(selected) - 5} more cases")

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "cases": selected}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")
    return selected


if __name__ == "__main__":
    main()
