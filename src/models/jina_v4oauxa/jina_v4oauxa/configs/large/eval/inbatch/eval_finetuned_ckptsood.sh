#!/bin/bash

set -e  # 遇到错误立即退出

# ============================================================================
# 微调Checkpoint评估脚本
# 评估使用不同reason_steps训练的微调checkpoint
# 
# 关键特性：
# 1. 复用现有的candidate pool embeddings (不重新编码)
# 2. 使用现有的mbeir_embedder.py和mbeir_retriever.py
# 3. 每个checkpoint使用独立的输出目录，避免路径冲突
# 4. 正确加载LoRA checkpoint
# ============================================================================

# --- 基础路径配置 ---
SRC="/data/LR1/src"
COMMON_DIR="$SRC/common"

UNIIR_BASE_DIR="/data/LR1"
MBEIR_DATA_DIR="/data/M-BEIR/"

# 配置文件目录
MODEL="jina_v4oauxa/jina_v4oauxa"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="eval"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs/$SIZE/$MODE/$EXP_NAME"

# 注意：不再使用共享的 cand_pool embeddings
# 每个 checkpoint 会用微调后的模型重新编码 candidates，保持训练-测试一致性

# --- 环境变量设置 ---
export CUDA_VISIBLE_DEVICES=0,1
NPROC=2
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 切换工作目录
cd $COMMON_DIR

# --- 微调Checkpoint配置 ---
# 格式: "名称:checkpoint路径:训练时reason_steps:评估时reason_steps"
#"MT0:/data/LR1/checkpointMT0/jina_v4o/Large/Instruct/InBatch/jina_v4o_step_600.pth:0:0"
#"MT1:/data/LR1/checkpointMT1/jina_v4o/Large/Instruct/InBatch/jina_v4o_emergency_epoch0_step0.pth:1:1"
#"MT3:/data/LR1/checkpointMT3/jina_v4o/Large/Instruct/InBatch/jina_v4o_step_100.pth:3:3"
#"MT0:/data/LR1/checkpointMTO0/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth:0:0"
# "MT1:/data/LR1/checkpointMTO1/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_0.pth:1:1"
# "MT3:/data/LR1/checkpointMTO3/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth:3:3"
# "MT5:/data/LR1/checkpointMTO5/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth:5:5"
CHECKPOINTS=(
    "MT0:/data/LR1/checkpointMTO0/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth:0:0"
    "MT5:/data/LR1/checkpointMTO5/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth:5:5"
)

# --- 结果汇总目录 ---
RESULTS_SUMMARY_DIR="$UNIIR_BASE_DIR/retrieval_results_summary/finetuned_ckpts"
mkdir -p $RESULTS_SUMMARY_DIR
LOG_FILE="$RESULTS_SUMMARY_DIR/evaluation_summary_$(date +%Y%m%d_%H%M%S).log"

echo "Fine-tuned Checkpoint Evaluation Summary - $(date)" > $LOG_FILE
echo "============================================================================" >> $LOG_FILE
echo "Note: Each checkpoint re-encodes candidates for train-test consistency" >> $LOG_FILE
echo "============================================================================" >> $LOG_FILE

# --- 自动清理机制 ---
TEMP_FILES=()
cleanup() {
    echo ""
    echo "Cleaning up temporary config files..."
    rm -f "${TEMP_FILES[@]}"
    echo "Cleanup done."
}
trap cleanup EXIT

# ============================================================================
# 主循环：遍历每个checkpoint
# ============================================================================
for CKPT_CONFIG in "${CHECKPOINTS[@]}"; do
    # 解析配置
    IFS=':' read -r CKPT_NAME CKPT_PATH TRAIN_RS EVAL_RS <<< "$CKPT_CONFIG"
    
    echo ""
    echo "############################################################################"
    echo "Evaluating: $CKPT_NAME"
    echo "  Checkpoint: $CKPT_PATH"
    echo "  Train reason_steps: $TRAIN_RS"
    echo "  Eval reason_steps: $EVAL_RS"
    echo "############################################################################"
    echo ""
    
    # 检查checkpoint文件是否存在
    if [ ! -f "$CKPT_PATH" ]; then
        echo "ERROR: Checkpoint not found: $CKPT_PATH"
        echo "Skipping $CKPT_NAME..."
        continue
    fi

    # 定义当前checkpoint的独立工作目录
    CURRENT_UNIIR_DIR="$UNIIR_BASE_DIR/runs/eval_finetuned/$CKPT_NAME"
    mkdir -p "$CURRENT_UNIIR_DIR"
    
    echo "Work Directory: $CURRENT_UNIIR_DIR"
    echo "" >> $LOG_FILE
    echo "Checkpoint: $CKPT_NAME" >> $LOG_FILE
    echo "  Path: $CKPT_PATH" >> $LOG_FILE
    echo "  Work Dir: $CURRENT_UNIIR_DIR" >> $LOG_FILE

    # 注意：为保持训练-测试一致性，每个 checkpoint 需要用微调后的模型重新编码 candidates
    # 不再使用预计算的 cand_pool embeddings（它们是用原始模型编码的）

    # 计算端口，防止端口冲突
    PORT_OFFSET=$(echo "$CKPT_NAME" | sed 's/MT//')
    export MASTER_PORT=$((29700 + PORT_OFFSET))
    export MASTER_ADDR=127.0.0.1

    # ------------------------------------------------------------------------
    # Step 1: Embedding (编码 query 和 cand_pool，使用微调后的模型)
    # ------------------------------------------------------------------------
    echo "--> Step 1: Embedding queries and candidates..."
    
    ORIG_EMBED_YAML="$CONFIG_DIR/embedood.yaml"
    TEMP_EMBED_YAML="$CONFIG_DIR/embed_${CKPT_NAME}_tmp.yaml"
    cp "$ORIG_EMBED_YAML" "$TEMP_EMBED_YAML"
    TEMP_FILES+=("$TEMP_EMBED_YAML")

    # 保持 cand_pool 编码启用（使用微调后的模型编码，保持训练-测试一致性）
    # 不修改 enable_embed 配置，使用 embed.yaml 中的默认值

    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TEMP_EMBED_YAML" \
        --enable_instruct True

    # 运行 Embedder (使用LoRA checkpoint)
    # 使用 --num_thought_tokens 替代已废弃的 --reason_steps
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port=$MASTER_PORT \
        --master_addr=$MASTER_ADDR \
        mbeir_embedder.py \
        --config_path "$TEMP_EMBED_YAML" \
        --uniir_dir "$CURRENT_UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --num_thought_tokens $EVAL_RS \
        --lora_checkpoint "$CKPT_PATH"

    # ------------------------------------------------------------------------
    # Step 2: Indexing
    # ------------------------------------------------------------------------
    echo "--> Step 2: Indexing..."
    
    ORIG_INDEX_YAML="$CONFIG_DIR/indexood.yaml"
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
        --num_thought_tokens $EVAL_RS

    # ------------------------------------------------------------------------
    # Step 3: Retrieval
    # ------------------------------------------------------------------------
    echo "--> Step 3: Retrieval..."

    ORIG_RETRIEVAL_YAML="$CONFIG_DIR/retrievalood.yaml"
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
        --num_thought_tokens $EVAL_RS 2>&1 | tee "$STEP_LOG"

    # 提取 Recall 指标
    echo "----------------------------------------" >> $LOG_FILE
    echo "Results for $CKPT_NAME (train_rs=$TRAIN_RS, eval_rs=$EVAL_RS):" >> $LOG_FILE
    grep -E "Mean Recall@|Recall@" "$STEP_LOG" >> $LOG_FILE || echo "No Recall metrics found" >> $LOG_FILE
    echo "" >> $LOG_FILE

    echo "Finished $CKPT_NAME"
done

# ============================================================================
# 最终汇总
# ============================================================================
echo ""
echo "############################################################################"
echo "All Checkpoint Evaluations Completed!"
echo "############################################################################"
echo "Summary Log: $LOG_FILE"
echo ""
echo "Individual Results:"
for CKPT_CONFIG in "${CHECKPOINTS[@]}"; do
    IFS=':' read -r CKPT_NAME CKPT_PATH TRAIN_RS EVAL_RS <<< "$CKPT_CONFIG"
    echo "  - $CKPT_NAME -> $UNIIR_BASE_DIR/runs/eval_finetuned/$CKPT_NAME/"
done
echo "############################################################################"

cat $LOG_FILE
