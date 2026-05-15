#!/bin/bash

set -e  # 遇到错误立即退出

# ============================================================================
# 微调前的模型；不同推理步骤
# 
# ============================================================================

# --- 基础路径配置 ---
SRC="/data/LR1/src"
COMMON_DIR="$SRC/common"

# 原始基础数据目录 (我们会在此目录下创建子文件夹)
UNIIR_BASE_DIR="/data/LR1"
MBEIR_DATA_DIR="/data/M-BEIR/sub_MBEIR/"

# 配置文件目录
MODEL="jina_v4/jina_v4"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="eval"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs/$SIZE/$MODE/$EXP_NAME"

# --- 环境变量设置 ---
export CUDA_VISIBLE_DEVICES=0,1
NPROC=2
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 切换工作目录
cd $COMMON_DIR

# --- 待测试参数 ---
# REASON_STEPS_LIST=(0 1 3)
REASON_STEPS_LIST=(0)
# --- 结果汇总目录 ---
RESULTS_SUMMARY_DIR="$UNIIR_BASE_DIR/retrieval_results_summary"
mkdir -p $RESULTS_SUMMARY_DIR
LOG_FILE="$RESULTS_SUMMARY_DIR/evaluation_summary_$(date +%Y%m%d_%H%M%S).log"

echo "Evaluation Summary - $(date)" > $LOG_FILE
echo "============================================================================" >> $LOG_FILE

# --- 自动清理机制 (Trap) ---
# 用于存储临时产生的配置文件路径，脚本退出时统一删除
TEMP_FILES=()
cleanup() {
    echo ""
    echo "Cleaning up temporary config files..."
    rm -f "${TEMP_FILES[@]}"
    echo "Cleanup done."
}
trap cleanup EXIT

# ============================================================================
# 主循环
# ============================================================================
for REASON_STEPS in "${REASON_STEPS_LIST[@]}"; do
    echo ""
    echo "############################################################################"
    echo "Starting Process for reason_steps=${REASON_STEPS}"
    echo "############################################################################"
    echo ""

    # [关键改进] 1. 定义当前轮次的独立工作目录
    # 所有生成的 Embedding 和 Index 都会强制存放在这个子目录下，互不干扰
    CURRENT_UNIIR_DIR="$UNIIR_BASE_DIR/runs/steps_${REASON_STEPS}"
    mkdir -p "$CURRENT_UNIIR_DIR"
    
    echo "Work Directory: $CURRENT_UNIIR_DIR"
    echo "Work Directory: $CURRENT_UNIIR_DIR" >> $LOG_FILE

    # 计算端口，防止端口冲突
    export MASTER_PORT=$((29610 + REASON_STEPS))
    export MASTER_ADDR=127.0.0.1

    # ------------------------------------------------------------------------
    # Step 1: Embedding
    # ------------------------------------------------------------------------
    echo "--> Step 1: Embedding..."
    
    # [关键改进] 2. 创建配置文件副本
    ORIG_EMBED_YAML="$CONFIG_DIR/embed.yaml"
    TEMP_EMBED_YAML="$CONFIG_DIR/embed_steps_${REASON_STEPS}_tmp.yaml"
    cp "$ORIG_EMBED_YAML" "$TEMP_EMBED_YAML"
    TEMP_FILES+=("$TEMP_EMBED_YAML") # 加入清理列表

    # 修改副本配置
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TEMP_EMBED_YAML" \
        --enable_instruct True

    # 运行 Embedder (传入独立的 CURRENT_UNIIR_DIR 和 副本配置)
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port=$MASTER_PORT \
        --master_addr=$MASTER_ADDR \
        mbeir_embedder.py \
        --config_path "$TEMP_EMBED_YAML" \
        --uniir_dir "$CURRENT_UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --reason_steps $REASON_STEPS

    # ------------------------------------------------------------------------
    # Step 2: Indexing
    # ------------------------------------------------------------------------
    echo "--> Step 2: Indexing..."
    
    ORIG_INDEX_YAML="$CONFIG_DIR/index.yaml"
    TEMP_INDEX_YAML="$CONFIG_DIR/index_steps_${REASON_STEPS}_tmp.yaml"
    cp "$ORIG_INDEX_YAML" "$TEMP_INDEX_YAML"
    TEMP_FILES+=("$TEMP_INDEX_YAML")

    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TEMP_INDEX_YAML" \
        --enable_instruct True

    # Indexing 会去读取 CURRENT_UNIIR_DIR 下刚才生成的 embedding
    python mbeir_retriever.py \
        --config_path "$TEMP_INDEX_YAML" \
        --uniir_dir "$CURRENT_UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --enable_create_index \
        --reason_steps $REASON_STEPS

    # ------------------------------------------------------------------------
    # Step 3: Retrieval
    # ------------------------------------------------------------------------
    echo "--> Step 3: Retrieval..."

    ORIG_RETRIEVAL_YAML="$CONFIG_DIR/retrieval.yaml"
    TEMP_RETRIEVAL_YAML="$CONFIG_DIR/retrieval_steps_${REASON_STEPS}_tmp.yaml"
    cp "$ORIG_RETRIEVAL_YAML" "$TEMP_RETRIEVAL_YAML"
    TEMP_FILES+=("$TEMP_RETRIEVAL_YAML")

    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TEMP_RETRIEVAL_YAML" \
        --enable_instruct True

    # 定义日志输出路径
    STEP_LOG="$RESULTS_SUMMARY_DIR/retrieval_log_steps_${REASON_STEPS}.txt"

    python mbeir_retriever.py \
        --config_path "$TEMP_RETRIEVAL_YAML" \
        --uniir_dir "$CURRENT_UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --enable_retrieval \
        --reason_steps $REASON_STEPS 2>&1 | tee "$STEP_LOG"

    # 提取 Recall 指标
    echo "----------------------------------------" >> $LOG_FILE
    echo "Results for reason_steps=${REASON_STEPS}:" >> $LOG_FILE
    grep -E "Mean Recall@" "$STEP_LOG" >> $LOG_FILE || echo "No Recall metrics found" >> $LOG_FILE
    echo "" >> $LOG_FILE

    echo "Finished reason_steps=${REASON_STEPS}"
done

# ============================================================================
# 最终汇总
# ============================================================================
echo ""
echo "############################################################################"
echo "All Evaluations Completed!"
echo "############################################################################"
echo "Summary Log: $LOG_FILE"
echo "Isolated Data Directories:"
for REASON_STEPS in "${REASON_STEPS_LIST[@]}"; do
    echo "  - steps=${REASON_STEPS} -> $UNIIR_BASE_DIR/runs/steps_${REASON_STEPS}/"
done
echo "############################################################################"