#!/bin/bash

set -e  # Exit on error

# ============================================================================
# Thought Token Initialization Test Script
# 
# Purpose: Test retrieval performance when thought_token embeddings are
# re-initialized to "." (English period) instead of using learned embeddings.
#
# Key features:
# 1. Loads LoRA weights from checkpoint (MT5 epoch 2)
# 2. Re-initializes thought token embeddings to "." (NOT from checkpoint)
# 3. Re-encodes queries only (cand_pool embeddings are symlinked)
# 4. Runs indexing and retrieval to compute final metrics
#
# This tests the contribution of learned thought tokens vs naive initialization.
# ============================================================================

# --- Basic path config ---
SRC="/data/LR1/src"
COMMON_DIR="$SRC/common"
UNIIR_BASE_DIR="/data/LR1"
MBEIR_DATA_DIR="/data/M-BEIR/sub_MBEIR/"

# Config file paths
MODEL="jina_v4od/jina_v4od"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="eval"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs/$SIZE/$MODE/$EXP_NAME"

# --- Experiment parameters ---
CKPT_NAME="MT5_thought_init_dot"
CKPT_PATH="/data/LR1/checkpointMTO5/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth"
NUM_THOUGHT_TOKENS=5
THOUGHT_INIT_TOKEN="."

# Source cand_pool embeddings to reuse
SOURCE_CAND_POOL="/data/LR1/runs/eval_finetuned/MtestT5/embed/JinaV4/Large/Instruct/InBatch/reason_steps_5/cand_pool"

# --- Output directory (unique, no conflicts) ---
CURRENT_UNIIR_DIR="$UNIIR_BASE_DIR/runs/eval_finetuned/${CKPT_NAME}"
EMBED_SUBDIR="embed/JinaV4/Large/Instruct/InBatch/reason_steps_${NUM_THOUGHT_TOKENS}"
TARGET_EMBED_DIR="$CURRENT_UNIIR_DIR/$EMBED_SUBDIR"
TARGET_CAND_POOL_DIR="$TARGET_EMBED_DIR/cand_pool"

# --- Environment ---
export CUDA_VISIBLE_DEVICES=2,6
NPROC=2
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd $COMMON_DIR

# --- Results directory ---
RESULTS_SUMMARY_DIR="$UNIIR_BASE_DIR/retrieval_results_summary/thought_init_test"
mkdir -p $RESULTS_SUMMARY_DIR
LOG_FILE="$RESULTS_SUMMARY_DIR/evaluation_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Thought Token Init Test - $(date)" > $LOG_FILE
echo "============================================================================" >> $LOG_FILE
echo "Checkpoint: $CKPT_PATH" >> $LOG_FILE
echo "Thought Init Token: '$THOUGHT_INIT_TOKEN'" >> $LOG_FILE
echo "Num Thought Tokens: $NUM_THOUGHT_TOKENS" >> $LOG_FILE
echo "Source Cand Pool: $SOURCE_CAND_POOL" >> $LOG_FILE
echo "Output Dir: $CURRENT_UNIIR_DIR" >> $LOG_FILE
echo "============================================================================" >> $LOG_FILE

# --- Cleanup ---
TEMP_FILES=()
cleanup() {
    echo ""
    echo "Cleaning up temporary config files..."
    rm -f "${TEMP_FILES[@]}"
    echo "Cleanup done."
}
trap cleanup EXIT

# --- Verify prerequisites ---
echo ""
echo "############################################################################"
echo "Thought Token Initialization Test"
echo "  Checkpoint: $CKPT_PATH"
echo "  Thought Init Token: '$THOUGHT_INIT_TOKEN'"
echo "  Num Thought Tokens: $NUM_THOUGHT_TOKENS"
echo "  Output Dir: $CURRENT_UNIIR_DIR"
echo "############################################################################"
echo ""

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

if [ ! -d "$SOURCE_CAND_POOL" ]; then
    echo "ERROR: Source cand_pool dir not found: $SOURCE_CAND_POOL"
    exit 1
fi

# --- Port config ---
export MASTER_PORT=29800
export MASTER_ADDR=127.0.0.1

# ============================================================================
# Step 0: Symlink candidate pool embeddings
# ============================================================================
echo "--> Step 0: Symlinking candidate pool embeddings..."

mkdir -p "$TARGET_EMBED_DIR"

# Create symlink for cand_pool (if not already exists)
if [ -L "$TARGET_CAND_POOL_DIR" ]; then
    echo "Symlink already exists: $TARGET_CAND_POOL_DIR -> $(readlink $TARGET_CAND_POOL_DIR)"
elif [ -d "$TARGET_CAND_POOL_DIR" ]; then
    echo "WARNING: $TARGET_CAND_POOL_DIR is a real directory. Skipping symlink."
else
    ln -s "$SOURCE_CAND_POOL" "$TARGET_CAND_POOL_DIR"
    echo "Created symlink: $TARGET_CAND_POOL_DIR -> $SOURCE_CAND_POOL"
fi

# Verify symlink
echo "Cand pool contents:"
ls -la "$TARGET_CAND_POOL_DIR/" | head -5
echo "..."

# ============================================================================
# Step 1: Embedding (queries only, with thought_token init to ".")
# ============================================================================
echo ""
echo "--> Step 1: Embedding queries (thought_token init='$THOUGHT_INIT_TOKEN')..."

EMBED_YAML="$CONFIG_DIR/embed_thought_init_test.yaml"

# Create a temporary copy with instruct enabled
TEMP_EMBED_YAML="$CONFIG_DIR/embed_${CKPT_NAME}_tmp.yaml"
cp "$EMBED_YAML" "$TEMP_EMBED_YAML"
TEMP_FILES+=("$TEMP_EMBED_YAML")

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$TEMP_EMBED_YAML" \
    --enable_instruct True

# Run the modified embedder that re-initializes thought tokens
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    mbeir_embedder_thought_init.py \
    --config_path "$TEMP_EMBED_YAML" \
    --uniir_dir "$CURRENT_UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --num_thought_tokens $NUM_THOUGHT_TOKENS \
    --lora_checkpoint "$CKPT_PATH" \
    --thought_init_token "$THOUGHT_INIT_TOKEN"

echo "Query embedding complete."

# ============================================================================
# Step 2: Indexing
# ============================================================================
echo ""
echo "--> Step 2: Indexing..."

ORIG_INDEX_YAML="$CONFIG_DIR/index.yaml"
TEMP_INDEX_YAML="$CONFIG_DIR/index_${CKPT_NAME}_tmp.yaml"
cp "$ORIG_INDEX_YAML" "$TEMP_INDEX_YAML"
TEMP_FILES+=("$TEMP_INDEX_YAML")

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$TEMP_INDEX_YAML" \
    --enable_instruct True

python mbeir_retriever.py \
    --config_path "$TEMP_INDEX_YAML" \
    --uniir_dir "$CURRENT_UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_create_index \
    --num_thought_tokens $NUM_THOUGHT_TOKENS

# ============================================================================
# Step 3: Retrieval
# ============================================================================
echo ""
echo "--> Step 3: Retrieval..."

ORIG_RETRIEVAL_YAML="$CONFIG_DIR/retrieval.yaml"
TEMP_RETRIEVAL_YAML="$CONFIG_DIR/retrieval_${CKPT_NAME}_tmp.yaml"
cp "$ORIG_RETRIEVAL_YAML" "$TEMP_RETRIEVAL_YAML"
TEMP_FILES+=("$TEMP_RETRIEVAL_YAML")

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$TEMP_RETRIEVAL_YAML" \
    --enable_instruct True

STEP_LOG="$RESULTS_SUMMARY_DIR/retrieval_log_${CKPT_NAME}.txt"

python mbeir_retriever.py \
    --config_path "$TEMP_RETRIEVAL_YAML" \
    --uniir_dir "$CURRENT_UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_retrieval \
    --num_thought_tokens $NUM_THOUGHT_TOKENS 2>&1 | tee "$STEP_LOG"

# ============================================================================
# Results
# ============================================================================
echo ""
echo "############################################################################"
echo "Thought Token Init Test Complete!"
echo "############################################################################"

echo "----------------------------------------" >> $LOG_FILE
echo "Results for ${CKPT_NAME}:" >> $LOG_FILE
echo "  Thought Init Token: '$THOUGHT_INIT_TOKEN'" >> $LOG_FILE
echo "  Num Thought Tokens: $NUM_THOUGHT_TOKENS" >> $LOG_FILE
grep -E "Mean Recall@|Recall@" "$STEP_LOG" >> $LOG_FILE || echo "No Recall metrics found" >> $LOG_FILE
echo "" >> $LOG_FILE

echo ""
echo "Summary Log: $LOG_FILE"
echo "Output Dir: $CURRENT_UNIIR_DIR"
echo ""
echo "Retrieval Results:"
grep -E "Mean Recall@|Recall@" "$STEP_LOG" || echo "No Recall metrics found in output"
echo ""
echo "############################################################################"

cat $LOG_FILE
