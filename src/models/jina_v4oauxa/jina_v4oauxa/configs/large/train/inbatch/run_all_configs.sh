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

MODEL="jina_v4oauxa"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="train"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/jina_v4oauxa/configs/$SIZE/$MODE/$EXP_NAME"
ORIGINAL_CONFIG="$CONFIG_DIR/inbatch.yaml"

# ============================================
# Resume / Skip Behavior
# ============================================
# If true: when final epoch checkpoint already exists, skip this config.
SKIP_COMPLETED=true

# If true: auto-detect latest checkpoint and resume training for each config.
AUTO_RESUME=true

# ============================================
# Stage-2 (hard-gate only) initialization
# ============================================
# When model.training_stage == 2, we always initialize from this Stage-1 checkpoint.
GATE_STAGE2_INIT_CKPT="/data/LR1/checkpoint/jina_v4oauxpro/Large/Instruct/InBatch_rs5_tt5/jina_v4oaux_epoch_2.pth"

# ============================================
# GPU and Environment Setup
# ============================================
export CUDA_VISIBLE_DEVICES=5,6
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
    "5 5"
    # "5 5"
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
            "$ckpt_dir"/jina_v4oaux_emergency_epoch*_step*.pth \
            "$ckpt_dir"/jina_v4oaux_step_*.pth \
            "$ckpt_dir"/jina_v4oaux_epoch_*.pth 2>/dev/null | head -n1 || true)
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
    CKPT_DIR="$UNIIR_DIR/checkpoint/jina_v4oaux/Large/Instruct/$EXP_SUFFIX"
    FINAL_EPOCH_CKPT="$CKPT_DIR/jina_v4oaux_epoch_${FINAL_EPOCH}.pth"

    # If this run is stage-2 hard-gate training, inject stage2_init_ckpt.
    TRAINING_STAGE=$(grep -E "^[[:space:]]*training_stage:[[:space:]]*[0-9]+" "$TEMP_CONFIG" | head -n1 | sed -E 's/.*training_stage:[[:space:]]*([0-9]+).*/\1/')
    if [[ -z "$TRAINING_STAGE" ]]; then
        TRAINING_STAGE=1
    fi

    if [[ "$TRAINING_STAGE" == "2" ]]; then
        if [[ ! -f "$GATE_STAGE2_INIT_CKPT" ]]; then
            echo "[ERROR] Stage-2 init checkpoint not found: $GATE_STAGE2_INIT_CKPT"
            exit 1
        fi

        if grep -q "^[[:space:]]*stage2_init_ckpt:" "$TEMP_CONFIG"; then
            sed -i "s#^\([[:space:]]*stage2_init_ckpt:\).*#\1 $GATE_STAGE2_INIT_CKPT#" "$TEMP_CONFIG"
        else
            sed -i "/^[[:space:]]*resume_training:/a\    stage2_init_ckpt: $GATE_STAGE2_INIT_CKPT" "$TEMP_CONFIG"
        fi

        echo "[Stage-2] hard-gate init checkpoint: $GATE_STAGE2_INIT_CKPT"
    fi

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
