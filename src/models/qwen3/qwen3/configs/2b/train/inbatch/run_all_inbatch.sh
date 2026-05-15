#!/bin/bash
#
# Qwen3-VL 批量自动化训练脚本
# 顺序执行配置矩阵: (num_thought_tokens, enable_final_token)

set -e

# ============================================
# 初始化虚拟环境
# ============================================
source /data/Qwen3-VL-Embedding/.venv/bin/activate

# ============================================
# 路径配置 (严格对齐 Jina 参考案例)
# ============================================
SRC="/data/LR1/src"
COMMON_DIR="/data/LR1/src/common"
UNIIR_DIR="/data/LR1"                  # 统一的输出目录
MBEIR_DATA_DIR="/data/M-BEIR"          # 统一的数据集目录

MODEL="qwen3"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="2b"
MODE="train"
EXP_BASE_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/qwen3/configs/$SIZE/$MODE/$EXP_BASE_NAME"
ORIGINAL_CONFIG="$CONFIG_DIR/inbatch.yaml"

# ============================================
# GPU 分配与环境设置
# ============================================
export CUDA_VISIBLE_DEVICES="4,5"   # <--- 请修改为你实际要使用的卡号
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export PYTHONPATH=$SRC

# ============================================
# 配置分布式环境变量
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
# 限制 CPU 线程，防止多进程下 CPU 爆炸与死锁
# ============================================
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# 将超时时间从 30 秒延长到 30 分钟 (1800秒)
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800

# 强制禁用 P2P 和 IB，这是 RTX 5090 在某些主板上死锁的常见原因
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# -----------------------------------
# NCCL 分布式通信调优 (防死锁)
# -----------------------------------
export NCCL_P2P_DISABLE=1          # 禁用点对点，强制走共享内存
export NCCL_IB_DISABLE=1           # 禁用 InfiniBand (如果你的机器只有网卡)
export NCCL_ALGO=Ring              # 强制使用环形算法同步梯度
export NCCL_ASYNC_ERROR_HANDLING=1 # 异步错误处理，防止一个进程崩了导致整机卡死
export TORCH_DISTRIBUTED_DEBUG=INFO # 开启分布式调试信息
# ============================================
# 批量实验矩阵配置
# ============================================
    # "0 0"
    # "0 1"
    # "4 1"
    # "2 1"
    #     "8 1"
    # "16 1"
    #     "4 1"
    # "0 1"
    # "2 1"
    # "8 1"
    # "16 1"
CONFIGS=(
"0 0"
)
BASE_PORT=29700

# ============================================
# 退出清理机制
# ============================================
TEMP_FILES=()
cleanup() {
    echo "🧹 清理所有生成的临时配置文件..."
    for f in "${TEMP_FILES[@]}"; do
        [ -f "$f" ] && rm -f "$f"
    done
}
trap cleanup EXIT

# ============================================
# 批量训练主循环
# ============================================
echo "========================================================"
echo " 🚀 开始自动化批量训练: 共 ${#CONFIGS[@]} 个配置"
echo " 📄 基础配置: $ORIGINAL_CONFIG"
echo " 🎮 使用显卡: [$CUDA_VISIBLE_DEVICES] (共 $NPROC 卡)"
echo " 📁 数据集路径: $MBEIR_DATA_DIR"
echo "========================================================"

RUN_IDX=0
for cfg in "${CONFIGS[@]}"; do
    TT=$(awk '{print $1}' <<< "$cfg")
    FINAL=$(awk '{print $2}' <<< "$cfg")
    RUN_IDX=$((RUN_IDX + 1))
    MASTER_PORT=$((BASE_PORT + RUN_IDX))

    EXP_SUFFIX="InBatch_tt${TT}_f${FINAL}"
    TEMP_CONFIG="$CONFIG_DIR/inbatch_tt${TT}_f${FINAL}.yaml"
    TEMP_FILES+=("$TEMP_CONFIG")

    echo ""
    echo "========================================================"
    echo " 🟢 正在启动实验 $RUN_IDX / ${#CONFIGS[@]}"
    echo "   ▶ num_thought_tokens = $TT"
    echo "   ▶ enable_final_token = $FINAL"
    echo "   ▶ exp_name          = $EXP_SUFFIX"
    echo "   ▶ master_port       = $MASTER_PORT"
    echo "========================================================"

    cp "$ORIGINAL_CONFIG" "$TEMP_CONFIG"

    sed -i "s/^[[:space:]]*num_thought_tokens:.*/  num_thought_tokens: $TT/" "$TEMP_CONFIG"
    sed -i "s/^[[:space:]]*mbeir_num_thought_tokens:.*/  mbeir_num_thought_tokens: $TT/" "$TEMP_CONFIG"
    if [ "$FINAL" = "1" ]; then
        sed -i "s/^[[:space:]]*enable_final_token:.*/  enable_final_token: true/" "$TEMP_CONFIG"
    else
        sed -i "s/^[[:space:]]*enable_final_token:.*/  enable_final_token: false/" "$TEMP_CONFIG"
    fi
    sed -i "s/^[[:space:]]*exp_name:.*/  exp_name: $EXP_SUFFIX/" "$TEMP_CONFIG"

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

    echo "✅ 实验 $RUN_IDX (TT=$TT, FINAL=$FINAL) 已顺利完成!"
    echo "--------------------------------------------------------"
    
    # 强制等待 3 秒释放显存
    sleep 3 
done

echo "🎉 恭喜！所有 ${#CONFIGS[@]} 个批量训练任务已全部结束！"