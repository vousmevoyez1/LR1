# """
# Main Analysis Pipeline: Thought Token Attention Analysis
# ========================================================
# End-to-end pipeline that:
# 1. Selects cases where MT5 (5 thought tokens) outperforms MT0 (0 thought tokens), or both models fail.
# 2. Loads the MT5 model checkpoint
# 3. Runs attention extraction on selected queries
# 4. Generates heatmap visualizations and quantitative metrics

# Usage:
#     # Full pipeline on a single dataset
#     python run_analysis.py --dataset webqa_task2 --selection_mode mt5_recall_mt0_miss

#     # Full pipeline on all M-BEIR query datasets
#     python run_analysis.py --all_datasets --selection_mode both_miss

#     # Skip case selection (use previously saved cases)
#     python run_analysis.py --skip_case_selection \
#         --cases_json output/selected_cases_all_union_mt5_recall_mt0_miss.json
# """

# import os
# import sys
# import json
# import argparse
# import time
# from typing import List, Dict, Optional

# import torch
# import numpy as np

# # Add project paths
# sys.path.insert(0, "/data/LR1/src")
# sys.path.insert(0, "/data/LR1/src/common")

# # ============================================================================
# # Constants
# # ============================================================================
# MBEIR_DATA_DIR = "/data/M-BEIR/sub_MBEIR"
# ANALYSIS_DIR = "/data/LR1/src/models/jina_v4o/analysis"
# OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "output")

# # Default checkpoints
# MT1_CHECKPOINT = "/data/LR1/checkpointMTO1/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_0.pth"
# MT5_CHECKPOINT = "/data/LR1/checkpointMTO5/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth"

# # Base model path
# BASE_MODEL_PATH = "/data/jina-v4-local-copy"


# # ============================================================================
# # Model Loading
# # ============================================================================

# def load_model_for_analysis(
#     checkpoint_path: str,
#     num_thought_tokens: int,
#     device: str = "cuda:0",
# ):
#     """
#     Load JinaEmbeddingsV4o model with LoRA checkpoint for attention analysis.
    
#     Args:
#         checkpoint_path: Path to the LoRA checkpoint .pth file.
#         num_thought_tokens: Number of thought tokens K.
#         device: Device to load model onto.
        
#     Returns:
#         (model, processor) tuple ready for inference.
#     """
#     from models.jina_v4o.jina_v4o.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model
    
#     print(f"Loading base model from: {BASE_MODEL_PATH}")
#     model = JinaEmbeddingsV4Model.from_pretrained(
#         BASE_MODEL_PATH,
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         attn_implementation="sdpa",  # Will fallback to eager when output_attentions=True
#     )
    
#     # Set up MBEIR attributes
#     model.mbeir_task_label = "retrieval"
#     model.mbeir_image_size = (224, 224)
#     model.mbeir_max_text_length = 512
#     model.mbeir_num_thought_tokens = num_thought_tokens
#     model.task = "retrieval"
    
#     # Set up thought tokens
#     if num_thought_tokens > 0:
#         model.setup_thought_tokens(num_thought_tokens, skip_init=True)
    
#     # Load LoRA checkpoint
#     if checkpoint_path and os.path.exists(checkpoint_path):
#         print(f"Loading LoRA checkpoint from: {checkpoint_path}")
#         from models.jina_v4o.train import load_lora_checkpoint
#         model, _ = load_lora_checkpoint(model, checkpoint_path)
#         print("Checkpoint loaded successfully!")
#     else:
#         print(f"WARNING: Checkpoint not found: {checkpoint_path}")
    
#     model.eval()
#     model = model.to(device)
    
#     processor = model.processor
    
#     print(f"Model loaded on {device} with {num_thought_tokens} thought tokens")
#     return model, processor


# # ============================================================================
# # Query Instruction Builder (matches training/eval pipeline)
# # ============================================================================

# def build_query_text_with_instruction(
#     query_entry: dict,
#     instructions_path: str = None,
# ) -> str:
#     """
#     Build the full query text with instruction prefix, matching the eval pipeline.
    
#     The eval pipeline prepends: "Retrieval Intent: {instruction}\nQuery: {text}"
    
#     Args:
#         query_entry: Dict from mbeir query JSONL with keys: query_txt, query_modality, task_id
#         instructions_path: Path to query_instructions.tsv
        
#     Returns:
#         Formatted query text string.
#     """
#     if instructions_path is None:
#         instructions_path = os.path.join(MBEIR_DATA_DIR, "instructions/query_instructions.tsv")
    
#     # Load instructions
#     instructions = {}
#     if os.path.exists(instructions_path):
#         with open(instructions_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split("\t")
#                 if len(parts) >= 5 and parts[0] != "query_modality":
#                     # key = (query_modality, cand_modality, dataset_name, dataset_id)
#                     key = (parts[0], parts[1], parts[2], parts[3])
#                     prompts = [p for p in parts[4:] if p.strip()]
#                     instructions[key] = prompts
    
#     query_txt = query_entry.get("query_txt") or ""
#     query_modality = query_entry.get("query_modality", "text")
#     task_id = query_entry.get("task_id", 0)
    
#     # Determine candidate modality and dataset from task_id
#     task_info = {
#         0: ("text", "image", "VisualNews", "0"),
#         1: ("text", "image", "Fashion200K", "1"),
#         2: ("text", "text", "WebQA", "2"),
#         3: ("image", "text", "MSCOCO", "9"),  # task3 maps to dataset 9
#         6: ("text,image", "text,image", "OVEN", "4"),
#     }
    
#     if task_id in task_info:
#         qm, cm, ds_name, ds_id = task_info[task_id]
#         key = (qm, cm, ds_name, ds_id)
#         prompts = instructions.get(key, [])
#         if prompts:
#             import random
#             instruction = prompts[0]  # Use first prompt consistently
#         else:
#             instruction = ""
#     else:
#         instruction = ""
    
#     # Format like the eval pipeline
#     if query_txt.strip():
#         formatted = f"Retrieval Intent: {instruction}\nQuery: {query_txt}"
#     else:
#         formatted = f"Retrieval Intent: {instruction}\nQuery: "
    
#     return formatted


# # ============================================================================
# # Analysis Pipeline
# # ============================================================================

# def run_attention_analysis(
#     cases: List[dict],
#     model,
#     processor,
#     num_thought_tokens: int,
#     device: str,
#     output_dir: str,
#     model_tag: str = "MT5",
#     num_cases_to_visualize: int = 10,
# ):
#     """
#     Run attention analysis on selected cases.
    
#     Args:
#         cases: List of case dicts from case_selector.
#         model: Loaded model.
#         processor: Model processor.
#         num_thought_tokens: K value.
#         device: Device string.
#         output_dir: Directory to save results.
#         model_tag: Identifier for the model (MT1/MT5).
#         num_cases_to_visualize: Number of cases to generate heatmaps for.
#     """
#     from attention_analyzer import (
#         AttentionExtractor,
#         compute_all_metrics,
#         plot_attention_heatmap,
#         plot_metrics_summary,
#         print_metrics_table,
#     )
    
#     extractor = AttentionExtractor(
#         model=model,
#         processor=processor,
#         num_thought_tokens=num_thought_tokens,
#         device=device,
#         task_label="retrieval",
#     )
    
#     viz_dir = os.path.join(output_dir, f"heatmaps_{model_tag}")
#     metrics_dir = os.path.join(output_dir, f"metrics_{model_tag}")
#     os.makedirs(viz_dir, exist_ok=True)
#     os.makedirs(metrics_dir, exist_ok=True)
    
#     all_metrics = []
#     num_to_viz = min(num_cases_to_visualize, len(cases))
    
#     print(f"\n{'='*70}")
#     print(f"  Running Attention Analysis: {model_tag} (K={num_thought_tokens})")
#     print(f"  Total cases: {len(cases)}, Visualizing: {num_to_viz}")
#     print(f"{'='*70}\n")
    
#     for idx, case in enumerate(cases):
#         qid = case["qid"]
#         query_txt = case.get("query_txt", "")
#         query_img_path = case.get("query_img_path", None)
        
#         # Build query text with instruction (matching eval pipeline)
#         formatted_query_txt = build_query_text_with_instruction(case)
        
#         print(f"[{idx+1}/{len(cases)}] Processing qid={qid} ...")
        
#         try:
#             # Extract attention
#             result = extractor.extract(
#                 query_text=formatted_query_txt if formatted_query_txt.strip() else None,
#                 query_image=None,
#                 query_image_path=query_img_path,
#             )
            
#             # Compute metrics
#             if result.thought_to_input_attn is not None:
#                 metrics = compute_all_metrics(result.thought_to_input_attn)
#                 metrics["qid"] = qid
#                 metrics["query_txt"] = query_txt[:100] if query_txt else ""
#                 metrics["query_modality"] = case.get("query_modality", "")
#                 metrics["delta_rank"] = case.get("delta_rank", 0)
#                 metrics["mt1_rank"] = case.get("mt1_rank", None)
#                 metrics["mt5_rank"] = case.get("mt5_rank", None)
#                 all_metrics.append(metrics)
                
#                 # Print metrics for this case
#                 print_metrics_table(metrics, qid=qid)
                
#                 # Generate visualizations for top cases
#                 if idx < num_to_viz:
#                     # Heatmap
#                     heatmap_path = os.path.join(viz_dir, f"heatmap_{qid.replace(':', '_')}.png")
#                     plot_attention_heatmap(
#                         result,
#                         save_path=heatmap_path,
#                         title=f"[{model_tag}] qid={qid} | ΔRank={case.get('delta_rank', '?')}",
#                     )
                    
#                     # Metrics plot
#                     metrics_path = os.path.join(metrics_dir, f"metrics_{qid.replace(':', '_')}.png")
#                     plot_metrics_summary(
#                         metrics,
#                         save_path=metrics_path,
#                         title=f"[{model_tag}] qid={qid} (K={num_thought_tokens})",
#                     )
#             else:
#                 print(f"  Skipping: no thought-to-input attention available.")
                
#         except Exception as e:
#             print(f"  ERROR processing qid={qid}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
        
#         # Clear GPU cache periodically
#         if (idx + 1) % 5 == 0:
#             torch.cuda.empty_cache()
    
#     # ================================================================
#     # Aggregate statistics across all cases
#     # ================================================================
#     if all_metrics:
#         print(f"\n{'='*70}")
#         print(f"  Aggregate Statistics ({model_tag}, K={num_thought_tokens})")
#         print(f"  Analyzed: {len(all_metrics)} cases")
#         print(f"{'='*70}")
        
#         # Average entropy per thought token position
#         K = num_thought_tokens
#         avg_entropy = np.zeros(K)
#         for m in all_metrics:
#             for k in range(min(K, len(m["entropy_per_thought"]))):
#                 avg_entropy[k] += m["entropy_per_thought"][k]
#         avg_entropy /= len(all_metrics)
        
#         print(f"\n  Average Entropy per Thought Token:")
#         for k in range(K):
#             print(f"    Thought_{k+1}: {avg_entropy[k]:.6f}")
        
#         # Average JSD
#         if K > 1:
#             avg_jsd = np.zeros(K - 1)
#             for m in all_metrics:
#                 for k in range(min(K - 1, len(m["jsd_between_adjacent"]))):
#                     avg_jsd[k] += m["jsd_between_adjacent"][k]
#             avg_jsd /= len(all_metrics)
            
#             print(f"\n  Average JSD between Adjacent Thought Tokens:")
#             for k in range(K - 1):
#                 print(f"    T{k+1} → T{k+2}: {avg_jsd[k]:.6f}")
#             print(f"    Overall Mean JSD: {np.mean(avg_jsd):.6f}")
        
#         # Count entropy trends
#         converging = sum(1 for m in all_metrics if m["entropy_trend"] == "converging")
#         diverging = len(all_metrics) - converging
#         print(f"\n  Entropy Trend: {converging} converging, {diverging} diverging")
        
#         # Save aggregate metrics
#         aggregate_path = os.path.join(output_dir, f"aggregate_metrics_{model_tag}.json")
#         with open(aggregate_path, "w") as f:
#             json.dump({
#                 "model_tag": model_tag,
#                 "num_thought_tokens": num_thought_tokens,
#                 "num_cases": len(all_metrics),
#                 "avg_entropy_per_thought": avg_entropy.tolist(),
#                 "avg_jsd_between_adjacent": avg_jsd.tolist() if K > 1 else [],
#                 "converging_count": converging,
#                 "diverging_count": diverging,
#                 "per_case_metrics": all_metrics,
#             }, f, indent=2, ensure_ascii=False)
#         print(f"\n  Saved aggregate metrics to: {aggregate_path}")
    
#     # Save all per-case metrics to a separate summary
#     summary_path = os.path.join(output_dir, f"per_case_summary_{model_tag}.json")
#     with open(summary_path, "w") as f:
#         json.dump(all_metrics, f, indent=2, ensure_ascii=False)
#     print(f"  Saved per-case summary to: {summary_path}")
    
#     return all_metrics


# # ============================================================================
# # Main Entry Point
# # ============================================================================

# def main():
#     parser = argparse.ArgumentParser(description="Thought Token Attention Analysis Pipeline")

#     # Case selection arguments
#     parser.add_argument("--dataset", type=str, default=None,
#                         help="Single dataset to analyze (ignored if --all_datasets)")
#     parser.add_argument("--all_datasets", action="store_true",
#                         help="Run on all M-BEIR query datasets")
#     parser.add_argument("--pool", type=str, default="union",
#                         choices=["union", "single"])
#     parser.add_argument("--selection_mode", type=str, default="mt5_recall_mt0_miss",
#                         choices=["mt5_recall_mt0_miss", "both_miss"],
#                         help="Selection mode: mt5_recall_mt0_miss or both_miss")
#     parser.add_argument("--top_n", type=int, default=None,
#                         help="Limit number of cases (None = all)")
#     parser.add_argument("--skip_case_selection", action="store_true",
#                         help="Skip case selection, use existing JSON")
#     parser.add_argument("--cases_json", type=str, default=None,
#                         help="Path to pre-saved cases JSON (with --skip_case_selection)")

#     # Model arguments
#     parser.add_argument("--model_tag", type=str, default="MT5",
#                         help="Model tag (MT1 or MT5)")
#     parser.add_argument("--num_thought_tokens", type=int, default=5,
#                         help="Number of thought tokens K")
#     parser.add_argument("--checkpoint", type=str, default=None,
#                         help="Path to LoRA checkpoint (auto-detected from model_tag if not set)")
#     parser.add_argument("--device", type=str, default="cuda:0")

#     # Analysis arguments
#     parser.add_argument("--num_cases_to_visualize", type=int, default=10,
#                         help="Number of cases to generate heatmaps for")
#     parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

#     args = parser.parse_args()

#     # Auto-detect checkpoint
#     if args.checkpoint is None:
#         if args.model_tag == "MT1":
#             args.checkpoint = MT1_CHECKPOINT
#         elif args.model_tag == "MT5":
#             args.checkpoint = MT5_CHECKPOINT
#         else:
#             raise ValueError(f"Unknown model_tag: {args.model_tag}. Set --checkpoint explicitly.")

#     os.makedirs(args.output_dir, exist_ok=True)

#     # ================================================================
#     # Step 1: Case Selection
#     # ================================================================
#     if args.skip_case_selection:
#         cases_path = args.cases_json
#         if cases_path is None:
#             if args.all_datasets:
#                 cases_path = os.path.join(
#                     args.output_dir,
#                     f"selected_cases_all_{args.pool}_{args.selection_mode}.json"
#                 )
#             else:
#                 dataset = args.dataset or "webqa_task2"
#                 cases_path = os.path.join(
#                     args.output_dir,
#                     f"selected_cases_{dataset}_{args.pool}_{args.selection_mode}.json"
#                 )
#         print(f"Loading pre-saved cases from: {cases_path}")
#         with open(cases_path, "r") as f:
#             data = json.load(f)
#             cases = data["cases"] if "cases" in data else data
#     else:
#         print("=" * 70)
#         print(f"  Step 1: Case Selection (mode={args.selection_mode})")
#         print("=" * 70)
#         sys.path.insert(0, ANALYSIS_DIR)
#         from case_selector import (
#             collect_cases_for_dataset,
#             collect_cases_for_all_datasets,
#             ALL_QUERY_DATASETS,
#         )

#         if args.all_datasets:
#             cases, metadata = collect_cases_for_all_datasets(
#                 datasets=ALL_QUERY_DATASETS,
#                 pool=args.pool,
#                 selection_mode=args.selection_mode,
#                 top_n=args.top_n,
#             )
#             cases_path = os.path.join(
#                 args.output_dir,
#                 f"selected_cases_all_{args.pool}_{args.selection_mode}.json"
#             )
#         else:
#             dataset = args.dataset or "webqa_task2"
#             cases, metadata = collect_cases_for_dataset(
#                 dataset=dataset,
#                 pool=args.pool,
#                 selection_mode=args.selection_mode,
#                 top_n=args.top_n,
#             )
#             cases_path = os.path.join(
#                 args.output_dir,
#                 f"selected_cases_{dataset}_{args.pool}_{args.selection_mode}.json"
#             )

#         with open(cases_path, "w", encoding="utf-8") as f:
#             json.dump({"metadata": metadata, "cases": cases}, f, indent=2, ensure_ascii=False)
#         print(f"Saved {len(cases)} cases to: {cases_path}")

#     print(f"\nSelected {len(cases)} cases for analysis.")

#     # ================================================================
#     # Step 2: Load Model
#     # ================================================================
#     print(f"\n{'='*70}")
#     print(f"  Step 2: Loading Model ({args.model_tag}, K={args.num_thought_tokens})")
#     print(f"{'='*70}")

#     model, processor = load_model_for_analysis(
#         checkpoint_path=args.checkpoint,
#         num_thought_tokens=args.num_thought_tokens,
#         device=args.device,
#     )

#     # ================================================================
#     # Step 3: Attention Analysis
#     # ================================================================
#     print(f"\n{'='*70}")
#     print(f"  Step 3: Attention Extraction & Analysis")
#     print(f"{'='*70}")

#     metrics = run_attention_analysis(
#         cases=cases,
#         model=model,
#         processor=processor,
#         num_thought_tokens=args.num_thought_tokens,
#         device=args.device,
#         output_dir=args.output_dir,
#         model_tag=args.model_tag,
#         num_cases_to_visualize=args.num_cases_to_visualize,
#     )

#     print(f"\n{'='*70}")
#     print(f"  Analysis Complete!")
#     print(f"  Output directory: {args.output_dir}")
#     print(f"{'='*70}")


# if __name__ == "__main__":
#     main()
"""
Main Analysis Pipeline: Thought Token Attention Analysis
========================================================
End-to-end pipeline that:
1. Loads the pre-selected cases where MT5 outperforms MT0.
2. Loads the MT5 model checkpoint.
3. Runs attention extraction grouped by sub-task.
4. Generates heatmap visualizations for the FIRST query of each task.
5. Computes and plots average metrics across all tasks.

Usage:
    # Run the batch analysis directly (defaults to the single pool json)
    python run_analysis.py --skip_case_selection
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Optional

import torch
import numpy as np

# Add project paths
sys.path.insert(0, "/data/LR1/src")
sys.path.insert(0, "/data/LR1/src/common")

# ============================================================================
# Constants
# ============================================================================
MBEIR_DATA_DIR = "/data/M-BEIR/sub_MBEIR"
ANALYSIS_DIR = "/data/LR1/src/models/jina_v4o/analysis"
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "outputjinao4")

# Default checkpoints (Only MT5 is needed based on requirements)
MT5_CHECKPOINT = "/data/LR1/checkpoint/jina_v4oauxpro/Large/Instruct/InBatch_rs5_tt5/jina_v4oaux_epoch_2.pth"
BASE_MODEL_PATH = "/data/jina-v4-local-copy"
DEFAULT_JSON_PATH = "/data/LR1/src/models/jina_v4o/analysis/output/selected_cases_all_union_mt5_recall_mt0_miss.json"


# ============================================================================
# Aggregate Visualization Function
# ============================================================================
def plot_subtask_aggregate_metrics(
    aggregated_data: dict,
    save_dir: str,
    title_suffix: str = ""
):
    """绘制所有子任务的总体平均注意力指标（熵和JSD）对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)
    tasks = list(aggregated_data.keys())
    
    if not tasks:
        print("No valid data to plot aggregates.")
        return

    colors = sns.color_palette("tab10", n_colors=len(tasks))
    
    # 1. 各子任务平均注意力熵趋势 (Line Plot)
    plt.figure(figsize=(10, 6))
    for i, task in enumerate(tasks):
        entropies = aggregated_data[task]["mean_entropies"]
        K = len(entropies)
        x_ticks = list(range(1, K + 1))
        plt.plot(x_ticks, entropies, marker='o', linewidth=2, 
                 color=colors[i], label=task)

    plt.xlabel("Thought Token Index", fontsize=12)
    plt.ylabel("Average Attention Entropy (nats)", fontsize=12)
    plt.title(f"Average Attention Entropy per Subtask {title_suffix}", fontsize=14, pad=15)
    plt.xticks(x_ticks, [f"T{i}" for i in x_ticks])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    entropy_save_path = os.path.join(save_dir, "aggregate_entropy_trend.png")
    plt.savefig(entropy_save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved aggregate entropy plot to: {entropy_save_path}")

    # 2. 各子任务平均注意力转移 JSD (Grouped Bar Chart)
    first_task = tasks[0]
    num_jsd_steps = len(aggregated_data[first_task]["mean_jsds"])
    
    if num_jsd_steps > 0:
        plt.figure(figsize=(12, 6))
        x = np.arange(num_jsd_steps) 
        width = 0.8 / len(tasks)
        
        for i, task in enumerate(tasks):
            jsds = aggregated_data[task]["mean_jsds"]
            offset = x - 0.4 + (i + 0.5) * width
            plt.bar(offset, jsds, width, label=task, color=colors[i], alpha=0.85)

        plt.xlabel("Adjacent Thought Token Pair", fontsize=12)
        plt.ylabel("Average Jensen-Shannon Divergence", fontsize=12)
        plt.title(f"Average Attention Shift (JSD) per Subtask {title_suffix}", fontsize=14, pad=15)
        
        step_labels = [f"T{i+1}→T{i+2}" for i in range(num_jsd_steps)]
        plt.xticks(x, step_labels)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        jsd_save_path = os.path.join(save_dir, "aggregate_jsd_shift.png")
        plt.savefig(jsd_save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved aggregate JSD plot to: {jsd_save_path}")


# ============================================================================
# Model Loading
# ============================================================================
def load_model_for_analysis(
    checkpoint_path: str,
    num_thought_tokens: int,
    device: str = "cuda:0",
):
    from models.jina_v4o.jina_v4o.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model
    
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    model = JinaEmbeddingsV4Model.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    
    model.mbeir_task_label = "retrieval"
    model.mbeir_image_size = (224, 224)
    model.mbeir_max_text_length = 512
    model.mbeir_num_thought_tokens = num_thought_tokens
    model.task = "retrieval"
    
    if num_thought_tokens > 0:
        model.setup_thought_tokens(num_thought_tokens, skip_init=True)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA checkpoint from: {checkpoint_path}")
        from models.jina_v4o.train import load_lora_checkpoint
        model, _ = load_lora_checkpoint(model, checkpoint_path)
        print("Checkpoint loaded successfully!")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    model = model.to(device)
    processor = model.processor
    
    print(f"Model loaded on {device} with {num_thought_tokens} thought tokens")
    return model, processor


# ============================================================================
# Query Instruction Builder
# ============================================================================
def build_query_text_with_instruction(query_entry: dict, instructions_path: str = None) -> str:
    if instructions_path is None:
        instructions_path = os.path.join(MBEIR_DATA_DIR, "instructions/query_instructions.tsv")
    
    instructions = {}
    if os.path.exists(instructions_path):
        with open(instructions_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5 and parts[0] != "query_modality":
                    key = (parts[0], parts[1], parts[2], parts[3])
                    prompts = [p for p in parts[4:] if p.strip()]
                    instructions[key] = prompts
    
    query_txt = query_entry.get("query_txt") or ""
    task_id = query_entry.get("task_id", 0)
    
    task_info = {
        0: ("text", "image", "VisualNews", "0"),
        1: ("text", "image", "Fashion200K", "1"),
        2: ("text", "text", "WebQA", "2"),
        3: ("image", "text", "MSCOCO", "9"),
        6: ("text,image", "text,image", "OVEN", "4"),
    }
    
    if task_id in task_info:
        qm, cm, ds_name, ds_id = task_info[task_id]
        key = (qm, cm, ds_name, ds_id)
        prompts = instructions.get(key, [])
        instruction = prompts[0] if prompts else ""
    else:
        instruction = ""
    
    if query_txt.strip():
        return f"Retrieval Intent: {instruction}\nQuery: {query_txt}"
    return f"Retrieval Intent: {instruction}\nQuery: "


# ============================================================================
# Analysis Pipeline
# ============================================================================
def run_attention_analysis(
    cases: List[dict],
    model,
    processor,
    num_thought_tokens: int,
    device: str,
    output_dir: str,
    model_tag: str = "MT5",
):
    from attention_analyzer import (
        AttentionExtractor,
        compute_all_metrics,
        plot_attention_heatmap,
        plot_metrics_summary,
    )
    
    extractor = AttentionExtractor(
        model=model,
        processor=processor,
        num_thought_tokens=num_thought_tokens,
        device=device,
        task_label="retrieval",
    )
    
    # 按任务分组整理案例
    cases_by_task = defaultdict(list)
    for case in cases:
        cases_by_task[case["dataset"]].append(case)

    aggregated_data = {}
    
    print(f"\n{'='*70}")
    print(f"  Running Attention Analysis: {model_tag} (K={num_thought_tokens})")
    print(f"  Found {len(cases_by_task)} tasks to process.")
    print(f"{'='*70}\n")
    
    for task, task_cases in cases_by_task.items():
        print(f"\n--- Processing Task: {task} ({len(task_cases)} cases) ---")
        
        # 为该任务创建独立的输出目录
        task_viz_dir = os.path.join(output_dir, f"heatmaps_{model_tag}", task)
        task_metrics_dir = os.path.join(output_dir, f"metrics_{model_tag}", task)
        os.makedirs(task_viz_dir, exist_ok=True)
        os.makedirs(task_metrics_dir, exist_ok=True)
        
        task_entropies = []
        task_jsds = []
        
        for idx, case in enumerate(task_cases):
            qid = case["qid"]
            query_txt = case.get("query_txt", "")
            query_img_path = case.get("query_img_path", None)
            formatted_query_txt = build_query_text_with_instruction(case)
            
            try:
                result = extractor.extract(
                    query_text=formatted_query_txt if formatted_query_txt.strip() else None,
                    query_image=None,
                    query_image_path=query_img_path,
                )
                
                if result.thought_to_input_attn is not None:
                    metrics = compute_all_metrics(result.thought_to_input_attn)
                    task_entropies.append(metrics["entropy_per_thought"])
                    if metrics["jsd_between_adjacent"]:
                        task_jsds.append(metrics["jsd_between_adjacent"])
                    
                    # 🌟 核心需求：只为该任务的第一条成功抽取的数据生成图表
                    if len(task_entropies) == 1:
                        print(f"  -> Generating plots for the first query in {task}: qid={qid}")
                        
                        heatmap_path = os.path.join(task_viz_dir, f"heatmap_{qid.replace(':', '_')}.png")
                        plot_attention_heatmap(
                            result, save_path=heatmap_path,
                            title=f"[{model_tag}] {task} | qid={qid}",
                        )
                        
                        
                        metrics_path = os.path.join(task_metrics_dir, f"metrics_{qid.replace(':', '_')}.png")
                        plot_metrics_summary(
                            metrics, save_path=metrics_path,
                            title=f"[{model_tag}] {task} | qid={qid}",
                        )
                else:
                    pass # 跳过无注意力矩阵的数据
                    
            except Exception as e:
                print(f"  ERROR processing {task} qid={qid}: {e}")
                continue
            
            # 定期清理显存
            if (idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

        # 计算该任务的平均值
        if task_entropies:
            mean_entropies = np.mean(task_entropies, axis=0).tolist()
            mean_jsds = np.mean(task_jsds, axis=0).tolist() if task_jsds else []
            aggregated_data[task] = {
                "mean_entropies": mean_entropies,
                "mean_jsds": mean_jsds
            }
            print(f"  Task {task} done. Valid cases: {len(task_entropies)}")

    # 循环结束，绘制所有任务的全局聚合对比图
    print(f"\n{'='*70}")
    print(f"  Generating Global Aggregate Metric Plots...")
    print(f"{'='*70}")
    plot_subtask_aggregate_metrics(
        aggregated_data=aggregated_data,
        save_dir=os.path.join(output_dir, f"aggregate_results_{model_tag}"),
        title_suffix="(MT5 Recall Cases)"
    )

    return aggregated_data


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Thought Token Attention Analysis Pipeline")

    # 默认直接跳过挑选并读取指定JSON
    parser.add_argument("--skip_case_selection", action="store_true", default=True,
                        help="Skip case selection, use existing JSON (default True)")
    parser.add_argument("--cases_json", type=str, default=DEFAULT_JSON_PATH,
                        help="Path to pre-saved cases JSON")

    # 模型参数绑定在 MT5 上
    parser.add_argument("--model_tag", type=str, default="MT5", help="Model tag (MT5)")
    parser.add_argument("--num_thought_tokens", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=MT5_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ================================================================
    # Step 1: Load pre-selected Cases
    # ================================================================
    cases_path = args.cases_json
    print(f"Loading pre-saved cases from: {cases_path}")
    with open(cases_path, "r") as f:
        data = json.load(f)
        cases = data.get("cases", data)
    print(f"Selected {len(cases)} cases for analysis.")

    # ================================================================
    # Step 2: Load Model
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Step 2: Loading Model ({args.model_tag}, K={args.num_thought_tokens})")
    print(f"{'='*70}")

    model, processor = load_model_for_analysis(
        checkpoint_path=args.checkpoint,
        num_thought_tokens=args.num_thought_tokens,
        device=args.device,
    )

    # ================================================================
    # Step 3: Attention Analysis (Grouped & Aggregated)
    # ================================================================
    run_attention_analysis(
        cases=cases,
        model=model,
        processor=processor,
        num_thought_tokens=args.num_thought_tokens,
        device=args.device,
        output_dir=args.output_dir,
        model_tag=args.model_tag,
    )

    print(f"\n{'='*70}")
    print(f"  Analysis Complete! Check out: {args.output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()