#!/bin/bash

set -e  # 遇到错误立即退出

# ============================================================================
# 微调Checkpoint评估脚本
# 评估使用不同reason_steps训练的微调checkpoint
# 
# 关键特性：
# 1. 每个checkpoint都重新编码candidate pool embeddings
# 2. 使用现有的mbeir_embedder.py和mbeir_retriever.py
# 3. 每个checkpoint使用独立的输出目录，避免路径冲突
# 4. 正确加载LoRA checkpoint
# 5. candidate pool采用不对称编码（reason_steps=0）
# ============================================================================

# --- 基础路径配置 ---
SRC="/data/LR1/src"
COMMON_DIR="$SRC/common"

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

# --- 微调Checkpoint配置 ---
# 格式: "名称:checkpoint路径:训练时reason_steps:评估时reason_steps"

#"MT3:/data/LR1/checkpoint/jina_v4/Large/Instruct/InBatch/jina_v4_epoch_0.pth:3:3"
CHECKPOINTS=(
    "MT3:/data/LR1/checkpointMT3all/jina_v4/Large/Instruct/InBatch/jina_v4_epoch_0.pth:3:3"
    "MT0:/data/LR1/checkpointMT0/jina_v4/Large/Instruct/InBatch/jina_v4_epoch_0.pth:0:0"
    
)

# --- 结果汇总目录 ---
RESULTS_SUMMARY_DIR="$UNIIR_BASE_DIR/retrieval_results_summary/finetuned_ckpts"
mkdir -p $RESULTS_SUMMARY_DIR
LOG_FILE="$RESULTS_SUMMARY_DIR/evaluation_summary_$(date +%Y%m%d_%H%M%S).log"

echo "Fine-tuned Checkpoint Evaluation Summary - $(date)" > $LOG_FILE
echo "============================================================================" >> $LOG_FILE
echo "Cand Pool Strategy: re-encode for each checkpoint (asymmetric query/candidate)" >> $LOG_FILE
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

    # 计算端口，防止端口冲突
    PORT_OFFSET=$(echo "$CKPT_NAME" | sed 's/MT//')
    export MASTER_PORT=$((29600 + PORT_OFFSET))
    export MASTER_ADDR=127.0.0.1

    # ------------------------------------------------------------------------
    # Step 1: Embedding (重新编码query和candidate pool)
    # ------------------------------------------------------------------------
    echo "--> Step 1: Embedding queries + candidate pools..."
    
    ORIG_EMBED_YAML="$CONFIG_DIR/embed.yaml"
    TEMP_EMBED_YAML="$CONFIG_DIR/embed_${CKPT_NAME}_tmp.yaml"
    cp "$ORIG_EMBED_YAML" "$TEMP_EMBED_YAML"
    TEMP_FILES+=("$TEMP_EMBED_YAML")

    # 修改配置：启用cand_pool重新编码 + 强制使用不对称query/candidate编码
    python -c "
import yaml
with open('$TEMP_EMBED_YAML', 'r') as f:
    config = yaml.safe_load(f)

# 启用cand_pool编码（不复用历史embedding）
config['embed_config']['cand_pools_config']['enable_embed'] = True

# 强制不对称编码：query用reason_steps，candidate用0
config.setdefault('model', {})
config['model']['symmetric_query_candidate_encoding'] = False

with open('$TEMP_EMBED_YAML', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Modified config: enabled cand_pool embedding + asymmetric query/candidate encoding')
"

    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TEMP_EMBED_YAML" \
        --enable_instruct True

    # 运行 Embedder (使用LoRA checkpoint)
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port=$MASTER_PORT \
        --master_addr=$MASTER_ADDR \
        mbeir_embedder.py \
        --config_path "$TEMP_EMBED_YAML" \
        --uniir_dir "$CURRENT_UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --reason_steps $EVAL_RS \
        --lora_checkpoint "$CKPT_PATH"

    # ------------------------------------------------------------------------
    # Step 2: Indexing
    # ------------------------------------------------------------------------
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
        --reason_steps $EVAL_RS

    # ------------------------------------------------------------------------
    # Step 3: Retrieval
    # ------------------------------------------------------------------------
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
        --reason_steps $EVAL_RS 2>&1 | tee "$STEP_LOG"

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
