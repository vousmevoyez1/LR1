#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Initialize Conda
source /root/anaconda3/etc/profile.d/conda.sh  # <--- Change this to your conda.sh path

# ============================================
# Path Configuration
# ============================================
SRC="/data/LR1/src"                    # Path to LR1/src
COMMON_DIR="/data/LR1/src/common"      # Path to common utilities
UNIIR_DIR="/data/LR1"                  # Directory for checkpoints, logs, etc.
MBEIR_DATA_DIR="/data/M-BEIR"          # MBEIR dataset directory

# Model and config paths
MODEL="jina_v4oaux"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="train"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/jina_v4oaux/configs/$SIZE/$MODE/$EXP_NAME"
CONFIG_PATH="$CONFIG_DIR/inbatch.yaml"

# ============================================
# GPU and Environment Setup
# ============================================
export CUDA_VISIBLE_DEVICES=3,4,5,6   # <--- Change to your GPU IDs
NPROC=4                                 # Number of GPUs
export PYTHONPATH=$SRC

echo "============================================"
echo "Training Configuration:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  CONFIG_PATH: $CONFIG_PATH"
echo "  MODEL_DIR: $MODEL_DIR"
echo "============================================"

# ============================================
# Optional: Update config for instruction mode
# ============================================
# cd $COMMON_DIR
# python config_updater.py \
#     --update_mbeir_yaml_instruct_status \
#     --mbeir_yaml_file_path $CONFIG_PATH \
#     --enable_instruct True

# ============================================
# Activate conda environment (if needed)
# ============================================
# conda activate your_env_name

# ============================================
# Run Distributed Training
# ============================================
cd $MODEL_DIR

export MASTER_PORT=29557
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# 增加 NCCL 超时时间到 60 分钟（默认是 10 分钟），防止长时间操作超时
export NCCL_TIMEOUT=3600
# 增加 PyTorch 分布式超时时间到 60 分钟（影响 init_process_group 的 timeout 默认值）
export TORCH_DISTRIBUTED_TIMEOUT=3600
# 设置 NCCL 阻塞等待超时（单位：毫秒），这是实际控制集合操作超时的关键参数
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1
# 启用 NCCL 调试信息
export NCCL_DEBUG=WARN
# 启用 FlightRecorder 以便调试
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    --rdzv_conf "timeout=3600" \
    train.py \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo "Training completed!"
