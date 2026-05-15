#!/bin/bash
#
# Automated training script for Jina V4 with different reason_steps / num_thought_tokens.
# Runs 4 configurations sequentially:
#   (reason_steps=0, num_thought_tokens=0)
#   (reason_steps=1, num_thought_tokens=1)
#   (reason_steps=3, num_thought_tokens=3)
#   (reason_steps=5, num_thought_tokens=5)
#
# Each run uses a unique exp_name to avoid checkpoint path conflicts.

set -e

# ============================================
# Path Configuration
# ============================================
SRC="/data/LR1/src"
COMMON_DIR="/data/LR1/src/common"
UNIIR_DIR="/data/LR1"
MBEIR_DATA_DIR="/data/M-BEIR"

MODEL="jina_v4od"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="train"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/jina_v4od/configs/$SIZE/$MODE/$EXP_NAME"
ORIGINAL_CONFIG="$CONFIG_DIR/inbatch.yaml"

# ============================================
# Resume / Skip Behavior
# ============================================
# If true: when final epoch checkpoint already exists, skip this config.
SKIP_COMPLETED=true

# If true: auto-detect latest checkpoint and resume training for each config.
AUTO_RESUME=true

# ============================================
# GPU and Environment Setup
# ============================================
export CUDA_VISIBLE_DEVICES=4,5
NPROC=2
export PYTHONPATH=$SRC

# Initialize Conda
source /root/anaconda3/etc/profile.d/conda.sh

# ============================================
# NCCL / Distributed Settings
# ============================================
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

# ============================================
# Configuration matrix: reason_steps num_thought_tokens
# ============================================
    # "0 0"
    # "1 1"
    # "3 3"
    # "5 5"
    #     "1 1"
    # "7 7"
    # "9 9"
        # "5 5"
    # "3 3"
CONFIGS=(
    "0 0"
    # "9 9"
    # "16 16"
    # "7 7"
)

# Base master port (incremented per run to avoid conflicts)
BASE_PORT=29565

# ============================================
# Helpers
# ============================================
get_num_train_epochs() {
    # Parse trainer_config.num_train_epochs from YAML (fallback=5)
    local n
    n=$(grep -E "^[[:space:]]*num_train_epochs:[[:space:]]*[0-9]+" "$ORIGINAL_CONFIG" \
        | head -n1 \
        | sed -E 's/.*num_train_epochs:[[:space:]]*([0-9]+).*/\1/')
    if [[ -z "$n" ]]; then
        n=5
    fi
    echo "$n"
}

pick_latest_checkpoint() {
    # Pick latest by mtime among emergency/step/epoch ckpts.
    local ckpt_dir="$1"
    local latest=""

    if [[ -d "$ckpt_dir" ]]; then
        latest=$(ls -1t \
            "$ckpt_dir"/jina_v4od_emergency_epoch*_step*.pth \
            "$ckpt_dir"/jina_v4od_step_*.pth \
            "$ckpt_dir"/jina_v4od_epoch_*.pth 2>/dev/null | head -n1 || true)
    fi

    echo "$latest"
}

# ============================================
# Cleanup on exit: remove temp configs
# ============================================
TEMP_FILES=()
cleanup() {
    for f in "${TEMP_FILES[@]}"; do
        [ -f "$f" ] && rm -f "$f"
    done
}
trap cleanup EXIT

# ============================================
# Main loop
# ============================================
echo "========================================================"
echo " Automated Training: ${#CONFIGS[@]} configurations"
echo " Original config: $ORIGINAL_CONFIG"
echo "========================================================"

NUM_EPOCHS=$(get_num_train_epochs)
FINAL_EPOCH=$((NUM_EPOCHS - 1))
echo " Parsed num_train_epochs = $NUM_EPOCHS (final epoch index = $FINAL_EPOCH)"

RUN_IDX=0
for cfg in "${CONFIGS[@]}"; do
    read -r RS TT <<< "$cfg"
    RUN_IDX=$((RUN_IDX + 1))
    MASTER_PORT=$((BASE_PORT + RUN_IDX))

    EXP_SUFFIX="InBatch_rs${RS}_tt${TT}"
    TEMP_CONFIG="$CONFIG_DIR/inbatch_rs${RS}_tt${TT}.yaml"
    TEMP_FILES+=("$TEMP_CONFIG")

    echo ""
    echo "========================================================"
    echo " Run $RUN_IDX / ${#CONFIGS[@]}"
    echo "   reason_steps     = $RS"
    echo "   num_thought_tokens = $TT"
    echo "   exp_name          = $EXP_SUFFIX"
    echo "   config            = $TEMP_CONFIG"
    echo "   master_port       = $MASTER_PORT"
    echo "========================================================"

    # Generate per-run config from the original
    cp "$ORIGINAL_CONFIG" "$TEMP_CONFIG"

    # Patch reason_steps, num_thought_tokens, and exp_name
    sed -i "s/^\(  reason_steps:\).*/\1 $RS/" "$TEMP_CONFIG"
    sed -i "s/^\(  num_thought_tokens:\).*/\1 $TT/" "$TEMP_CONFIG"
    sed -i "s/^\(  exp_name:\).*/\1 $EXP_SUFFIX/" "$TEMP_CONFIG"

    # Resolve checkpoint directory for this exp_name
    # Derived from inbatch.yaml path_suffix template:
    # checkpoint/${model.short_name}/${model.size}/${experiment.instruct_status}/${experiment.exp_name}/
    CKPT_DIR="$UNIIR_DIR/checkpoint/jina_v4od/Large/Instruct/$EXP_SUFFIX"
    FINAL_EPOCH_CKPT="$CKPT_DIR/jina_v4od_epoch_${FINAL_EPOCH}.pth"

    # Skip already completed configuration
    if [[ "$SKIP_COMPLETED" == "true" && -f "$FINAL_EPOCH_CKPT" ]]; then
        echo "[SKIP] Found final checkpoint: $FINAL_EPOCH_CKPT"
        echo "Run $RUN_IDX (rs=$RS, tt=$TT) skipped (already completed)."
        echo ""
        continue
    fi

    # Auto-resume from latest checkpoint if available
    if [[ "$AUTO_RESUME" == "true" ]]; then
        LATEST_CKPT=$(pick_latest_checkpoint "$CKPT_DIR")
        if [[ -n "$LATEST_CKPT" ]]; then
            LATEST_CKPT_BN=$(basename "$LATEST_CKPT")
            sed -i "s/^\(    resume_training:\).*/\1 true/" "$TEMP_CONFIG"
            sed -i "s/^\(    ckpt_name:\).*/\1 $LATEST_CKPT_BN/" "$TEMP_CONFIG"
            echo "[RESUME] Using checkpoint: $LATEST_CKPT"
        else
            sed -i "s/^\(    resume_training:\).*/\1 false/" "$TEMP_CONFIG"
            echo "[RESUME] No checkpoint found under: $CKPT_DIR"
            echo "         Start training from scratch for this config."
        fi
    fi

    echo "--- Generated config diff ---"
    diff "$ORIGINAL_CONFIG" "$TEMP_CONFIG" || true
    echo "-----------------------------"

    # Run training
    cd "$MODEL_DIR"
    export MASTER_PORT

    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port=$MASTER_PORT \
        --master_addr=$MASTER_ADDR \
        --rdzv_conf "timeout=3600" \
        train.py \
        --config_path "$TEMP_CONFIG" \
        --uniir_dir "$UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"

    echo "Run $RUN_IDX (rs=$RS, tt=$TT) completed!"
    echo ""
done

echo "========================================================"
echo " All ${#CONFIGS[@]} training runs completed!"
echo "========================================================"
