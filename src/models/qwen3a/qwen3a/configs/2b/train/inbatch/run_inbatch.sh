#!/bin/bash
set -e  # 遇到错误立即退出

# ============================================
# 初始化虚拟环境 (使用 Qwen3 的 uv 虚拟环境)
# ============================================
source /data/Qwen3-VL-Embedding/.venv/bin/activate

# ============================================
# 路径与模型配置 (与 Jina 优秀案例严格一致)
# ============================================
SRC="/data/LR1/src"
COMMON_DIR="/data/LR1/src/common"
UNIIR_DIR="/data/LR1"                  # 统一的输出目录
MBEIR_DATA_DIR="/data/M-BEIR"          # 统一的数据集目录

MODEL="qwen3a/qwen3a"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="2b"
MODE="train"
EXP_BASE_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/qwen3a/configs/$SIZE/$MODE/$EXP_BASE_NAME"
ORIGINAL_CONFIG="$CONFIG_DIR/inbatch.yaml"

# ============================================
# 动态运行参数配置 (支持外部传参)
# 用法: bash run_inbatch.sh [K] [FINAL] [EPOCHS] [DEBUG]
# 例子: bash run_inbatch.sh 5 1 3 1
# ============================================
TT=${1:-5}                  # reason steps K (当前实现中由 num_thought_tokens 驱动)
FINAL=${2:-1}               # enable_final_token: 1=true, 0=false（递归模式下通常可忽略）
EPOCHS=${3:-3}              # 训练 epoch 数
DEBUG=${4:-1}               # 1=开启 debug, 0=关闭 debug
EXP_SUFFIX="InBatch_k${TT}_f${FINAL}_e${EPOCHS}_dbg${DEBUG}"
CONFIG_PATH="$CONFIG_DIR/inbatch_k${TT}_f${FINAL}_e${EPOCHS}_dbg${DEBUG}.yaml"

# ============================================
# GPU 分配与环境设置
# ============================================
export CUDA_VISIBLE_DEVICES="0,1,2,3"   # <--- 请修改为你实际要使用的卡号
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export PYTHONPATH=$SRC

echo "============================================"
echo "🚀 启动 Qwen3-VL 分布式训练:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  分配进程数 (NPROC): $NPROC"
echo "  Thought Tokens 数量 (TT): $TT"
echo "  Final Token 开关 (FINAL): $FINAL"
echo "  Epoch 数量 (EPOCHS): $EPOCHS"
echo "  Debug 模式 (DEBUG): $DEBUG"
echo "  实验名称后缀: $EXP_SUFFIX"
echo "  数据路径: $MBEIR_DATA_DIR"
echo "============================================"

# ============================================
# 动态生成本次运行的专属 YAML 配置
# ============================================
cp "$ORIGINAL_CONFIG" "$CONFIG_PATH"

sed -i "s/^[[:space:]]*num_thought_tokens:.*/  num_thought_tokens: $TT/" "$CONFIG_PATH"
sed -i "s/^[[:space:]]*mbeir_num_thought_tokens:.*/  mbeir_num_thought_tokens: $TT/" "$CONFIG_PATH"
sed -i "s/^[[:space:]]*reason_steps:.*/  reason_steps: $TT/" "$CONFIG_PATH"
sed -i "s/^[[:space:]]*num_train_epochs:.*/  num_train_epochs: $EPOCHS/" "$CONFIG_PATH"
sed -i "s/^[[:space:]]*use_recursive_embedding_reasoning:.*/  use_recursive_embedding_reasoning: true/" "$CONFIG_PATH"
if [ "$FINAL" = "1" ]; then
    sed -i "s/^[[:space:]]*enable_final_token:.*/  enable_final_token: true/" "$CONFIG_PATH"
else
    sed -i "s/^[[:space:]]*enable_final_token:.*/  enable_final_token: false/" "$CONFIG_PATH"
fi
if [ "$DEBUG" = "1" ]; then
    sed -i "s/^[[:space:]]*debug_mode:.*/  debug_mode: true/" "$CONFIG_PATH"
    sed -i "s/^[[:space:]]*debug_token_preview_len:.*/  debug_token_preview_len: 32/" "$CONFIG_PATH"
    sed -i "s/^[[:space:]]*debug_print_freq:.*/  debug_print_freq: 1/" "$CONFIG_PATH"
else
    sed -i "s/^[[:space:]]*debug_mode:.*/  debug_mode: false/" "$CONFIG_PATH"
fi
sed -i "s/^[[:space:]]*exp_name:.*/  exp_name: $EXP_SUFFIX/" "$CONFIG_PATH"

# 退出时自动清理临时配置文件
cleanup() {
    [ -f "$CONFIG_PATH" ] && rm -f "$CONFIG_PATH"
    echo "🧹 已清理临时配置文件: $CONFIG_PATH"
}
trap cleanup EXIT

# ============================================
# 配置分布式环境变量 (防卡死与超时配置)
# ============================================
cd $MODEL_DIR

export MASTER_PORT=$((29000 + RANDOM % 1000)) 
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
# 启动训练
# ============================================
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    --rdzv_conf "timeout=3600" \
    train.py \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo "🎉 训练任务已结束！"