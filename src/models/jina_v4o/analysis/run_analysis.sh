#!/bin/bash
# ============================================================================
# Thought Token Attention Analysis — Launcher Script
# ============================================================================
# This script runs the full analysis pipeline:
#   1) Case selection: Find MT5-recall / MT0-miss or both-miss queries
#   2) Attention analysis: Extract & visualize thought token attention patterns
#
# Usage:
#   bash run_analysis.sh                                    # Single dataset (webqa_task2), mt5_recall_mt0_miss
#   bash run_analysis.sh --all_datasets                     # All M-BEIR datasets, mt5_recall_mt0_miss
#   bash run_analysis.sh --all_datasets --selection_mode both_miss
#   bash run_analysis.sh --dataset mscoco_task3 --selection_mode both_miss
# ============================================================================

set -e

# --- Default Configuration ---
DATASET=""
ALL_DATASETS=""
SELECTION_MODE="mt5_recall_mt0_miss"
MODEL_TAG="MT5"
NUM_VIZ=10
TOP_N=""

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --all_datasets)
            ALL_DATASETS="--all_datasets"
            shift
            ;;
        --selection_mode)
            SELECTION_MODE="$2"
            shift 2
            ;;
        --model_tag)
            MODEL_TAG="$2"
            shift 2
            ;;
        --num_viz)
            NUM_VIZ="$2"
            shift 2
            ;;
        --top_n)
            TOP_N="--top_n $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

ANALYSIS_DIR="/data/LR1/src/models/jina_v4o/analysis"
OUTPUT_DIR="$ANALYSIS_DIR/output"
SRC_DIR="/data/LR1/src"

export PYTHONPATH="$SRC_DIR:$ANALYSIS_DIR"
export CUDA_VISIBLE_DEVICES=6

# Determine K based on model tag
if [ "$MODEL_TAG" == "MT1" ]; then
    NUM_THOUGHT_TOKENS=1
    CHECKPOINT="/data/LR1/checkpointMTO1/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_0.pth"
elif [ "$MODEL_TAG" == "MT5" ]; then
    NUM_THOUGHT_TOKENS=5
    CHECKPOINT="/data/LR1/checkpointMTO5/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth"
else
    echo "ERROR: Unknown model tag '$MODEL_TAG'. Use MT1 or MT5."
    exit 1
fi

# Determine cases file name
if [ -n "$ALL_DATASETS" ]; then
    CASES_FILE="$OUTPUT_DIR/selected_cases_all_union_${SELECTION_MODE}.json"
    DATASET_ARG=""
    LOG_SUFFIX="all_${SELECTION_MODE}"
else
    DATASET="${DATASET:-webqa_task2}"
    CASES_FILE="$OUTPUT_DIR/selected_cases_${DATASET}_union_${SELECTION_MODE}.json"
    DATASET_ARG="--dataset $DATASET"
    LOG_SUFFIX="${DATASET}_${SELECTION_MODE}"
fi

echo "============================================================================"
echo "  Thought Token Attention Analysis"
echo "============================================================================"
echo "  Selection Mode:    $SELECTION_MODE"
if [ -n "$ALL_DATASETS" ]; then
    echo "  Datasets:          ALL M-BEIR query datasets"
else
    echo "  Dataset:           $DATASET"
fi
echo "  Model:             $MODEL_TAG (K=$NUM_THOUGHT_TOKENS)"
echo "  Checkpoint:        $CHECKPOINT"
echo "  Output:            $OUTPUT_DIR"
echo "  Cases to visualize: $NUM_VIZ"
echo "============================================================================"

mkdir -p "$OUTPUT_DIR"
cd "$ANALYSIS_DIR"

# --- Check if cases file exists ---
if [ -f "$CASES_FILE" ]; then
    echo ""
    echo "Cases file already exists: $CASES_FILE"
    echo "Skipping case selection. Delete it to re-run."
    SKIP_FLAG="--skip_case_selection --cases_json $CASES_FILE"
else
    echo ""
    echo "Running case selection..."
    SKIP_FLAG=""
fi

# --- Run Analysis ---
python run_analysis.py \
    $ALL_DATASETS \
    $DATASET_ARG \
    --pool union \
    --selection_mode "$SELECTION_MODE" \
    $TOP_N \
    --model_tag "$MODEL_TAG" \
    --num_thought_tokens $NUM_THOUGHT_TOKENS \
    --checkpoint "$CHECKPOINT" \
    --num_cases_to_visualize $NUM_VIZ \
    --output_dir "$OUTPUT_DIR" \
    $SKIP_FLAG \
    2>&1 | tee "$OUTPUT_DIR/analysis_log_${MODEL_TAG}_${LOG_SUFFIX}.txt"

echo ""
echo "============================================================================"
echo "  Analysis complete! Results in: $OUTPUT_DIR"
echo "============================================================================"
echo "  Heatmaps:  $OUTPUT_DIR/heatmaps_${MODEL_TAG}/"
echo "  Metrics:   $OUTPUT_DIR/metrics_${MODEL_TAG}/"
echo "  Summary:   $OUTPUT_DIR/aggregate_metrics_${MODEL_TAG}.json"
echo "============================================================================"
