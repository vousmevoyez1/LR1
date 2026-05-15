#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# ==========================================
# 1. 激活我们配置好的 uv 虚拟环境 (关键修改!)
# ==========================================
# echo "Activating virtual environment..."
# source /data/Qwen3-VL-Embedding/.venv/bin/activate

# ==========================================
# 2. 路径配置
# ==========================================
SRC="/data/LR1/src"  
COMMON_DIR="$SRC/common"
UNIIR_DIR="/data/LR1" 
MBEIR_DATA_DIR="/data/M-BEIR/sub_MBEIR/"  

MODEL="qwen3a/qwen3a"  
MODEL_DIR="$SRC/models/$MODEL"
SIZE="2b"
MODE="eval"  
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs/$SIZE/$MODE/$EXP_NAME"

# ==========================================
# 3. 多卡与环境变量配置 (关键修改!)
# ==========================================
# 假设你想用卡 0,1,2,3 进行 4 卡并行，请按需修改
export CUDA_VISIBLE_DEVICES=7
# 限制底层数学库的 CPU 线程数，防止多进程环境下的线程爆炸
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
NPROC=1

# 必须同时包含 LR1 和 Qwen3 的源码路径，防止找不到包！
export PYTHONPATH=$SRC:/data/Qwen3-VL-Embedding/src

echo "=========================================="
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NPROC (GPUs): $NPROC"
echo "=========================================="

cd $COMMON_DIR

# ==========================================
# 4. 运行 Embedding (多卡 DDP)
# ==========================================
CONFIG_PATH="$CONFIG_DIR/embed.yaml"
SCRIPT_NAME="mbeir_embedder.py"
echo -e "\n>>> Starting Embedding Phase: $CONFIG_PATH"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

export MASTER_PORT=29560
export MASTER_ADDR=127.0.0.1

# 使用 torchrun 启动多卡并行
torchrun \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    /data/LR1/src/common/mbeir_embedder.py \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

# ==========================================
# 5. 运行 Index (通常为单进程)
# ==========================================
CONFIG_PATH="$CONFIG_DIR/index.yaml"
SCRIPT_NAME="mbeir_retriever.py"
echo -e "\n>>> Starting Indexing Phase: $CONFIG_PATH"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

python $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_create_index

# ==========================================
# 6. 运行 Retrieval (检索与打分)
# ==========================================
CONFIG_PATH="$CONFIG_DIR/retrieval.yaml"
SCRIPT_NAME="mbeir_retriever.py"
echo -e "\n>>> Starting Retrieval Phase: $CONFIG_PATH"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

python $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_retrieval

echo -e "\n=========================================="
echo "All Evaluation Phases Completed Successfully!"
echo "=========================================="