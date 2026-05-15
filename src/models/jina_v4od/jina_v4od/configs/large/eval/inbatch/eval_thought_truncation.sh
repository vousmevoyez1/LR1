#!/bin/bash

set -e

# ============================================================================
# 截断推理评估脚本 (Truncated Reasoning Evaluation)
#
# 对训练了K个thought_token的模型，分别提取每个thought_token位置的embedding，
# 评估检索性能随推理步骤的变化。
#
# 优化设计：
# 1. 一次 forward pass 同时提取所有 K 个 thought token 的 embedding
# 2. Candidate pool 复用已有编码结果，通过 symlink 共享
# 3. 仅 indexing 和 retrieval 需要对每个 step 分别执行
# ============================================================================

# --- 基础路径配置 ---
SRC="/data/LR1/src"
COMMON_DIR="$SRC/common"

UNIIR_BASE_DIR="/data/LR1"
MBEIR_DATA_DIR="/data/M-BEIR/sub_MBEIR/"

# 配置文件目录
MODEL="jina_v4od/jina_v4od"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="eval"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs/$SIZE/$MODE/$EXP_NAME"

# --- 环境变量设置 ---
export CUDA_VISIBLE_DEVICES=4,5
NPROC=2
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 切换工作目录
cd $COMMON_DIR

# --- 模型配置 ---
CKPT_NAME="MTtrunnewlossO5"
CKPT_PATH="/data/LR1/checkpoint/jina_v4onewloss/Large/Instruct/InBatch_rs5_tt5/jina_v4o_epoch_1.pth"
NUM_THOUGHT_TOKENS=5   # 训练时使用的thought token数量

# --- 结果汇总目录 ---
RESULTS_SUMMARY_DIR="$UNIIR_BASE_DIR/retrieval_results_summary/thought_truncation_${CKPT_NAME}"
mkdir -p $RESULTS_SUMMARY_DIR
LOG_FILE="$RESULTS_SUMMARY_DIR/evaluation_summary_$(date +%Y%m%d_%H%M%S).log"

echo "Thought Token Truncation Evaluation - $(date)" > $LOG_FILE
echo "============================================================================" >> $LOG_FILE
echo "Checkpoint: $CKPT_NAME ($CKPT_PATH)" >> $LOG_FILE
echo "Num thought tokens: $NUM_THOUGHT_TOKENS" >> $LOG_FILE
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

# --- 检查checkpoint ---
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

# --- 定义工作目录 ---
BASE_WORK_DIR="$UNIIR_BASE_DIR/runs/eval_thought_truncation/$CKPT_NAME"
mkdir -p "$BASE_WORK_DIR"

# --- 复用已有的 candidate pool embeddings ---
EXISTING_CAND_DIR="/data/LR1/runs/eval_finetuned/MnewlossT5/embed/JinaV4/Large/Instruct/InBatch/reason_steps_5/cand_pool"

echo "Base work directory: $BASE_WORK_DIR"
echo "Reusing existing cand_pool: $EXISTING_CAND_DIR"

if [ ! -d "$EXISTING_CAND_DIR" ]; then
    echo "ERROR: Existing cand_pool directory not found: $EXISTING_CAND_DIR"
    exit 1
fi

ORIG_EMBED_YAML="$CONFIG_DIR/embed.yaml"

# --- 端口配置 ---
BASE_PORT=29800
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$BASE_PORT

# ============================================================================
# 第一步：一次编码，同时提取所有 K 个 thought step 的 query embedding
# ============================================================================
echo ""
echo "############################################################################"
echo "Step 1: Encoding ALL ${NUM_THOUGHT_TOKENS} thought steps in a single pass"
echo "############################################################################"

TEMP_EMBED_YAML="$CONFIG_DIR/embed_truncation_all_steps_tmp.yaml"
cp "$ORIG_EMBED_YAML" "$TEMP_EMBED_YAML"
TEMP_FILES+=("$TEMP_EMBED_YAML")

# 修改配置：只编码query，不编码candidate pool
python -c "
import yaml
with open('$TEMP_EMBED_YAML', 'r') as f:
    config = yaml.safe_load(f)

# 禁用 candidate pool 编码（复用已有结果）
config['embed_config']['cand_pools_config']['enable_embed'] = False
# 启用 query 编码
config['embed_config']['test_datasets_config']['enable_embed'] = True

with open('$TEMP_EMBED_YAML', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$TEMP_EMBED_YAML" \
    --enable_instruct True

# 一次编码，--all_steps 模式会将结果保存到 step_1/, step_2/, ..., step_K/ 子目录
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    mbeir_embedder_per_step.py \
    --config_path "$TEMP_EMBED_YAML" \
    --uniir_dir "$BASE_WORK_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --num_thought_tokens $NUM_THOUGHT_TOKENS \
    --all_steps \
    --lora_checkpoint "$CKPT_PATH"

echo "All query embeddings encoded successfully!"

# ============================================================================
# 第二步：为每个 step 创建 candidate pool symlink
# ============================================================================
echo ""
echo "############################################################################"
echo "Step 2: Creating candidate pool symlinks for all steps"
echo "############################################################################"

PATH_SUFFIX="JinaV4/Large/Instruct/InBatch/reason_steps_${NUM_THOUGHT_TOKENS}"

for STEP in $(seq 1 $NUM_THOUGHT_TOKENS); do
    STEP_CAND_DIR="$BASE_WORK_DIR/step_${STEP}/embed/$PATH_SUFFIX/cand_pool"

    if [ ! -e "$STEP_CAND_DIR" ]; then
        mkdir -p "$(dirname "$STEP_CAND_DIR")"
        ln -s "$EXISTING_CAND_DIR" "$STEP_CAND_DIR"
        echo "  Symlinked: step_${STEP}/cand_pool -> $EXISTING_CAND_DIR"
    else
        echo "  Already exists: step_${STEP}/cand_pool"
    fi
done

# ============================================================================
# 第三步：对每个 step 执行 indexing + retrieval
# ============================================================================
for STEP in $(seq 1 $NUM_THOUGHT_TOKENS); do
    echo ""
    echo "############################################################################"
    echo "Step 3.${STEP}: Indexing + Retrieval for thought_step=${STEP} / ${NUM_THOUGHT_TOKENS}"
    echo "############################################################################"

    STEP_WORK_DIR="$BASE_WORK_DIR/step_${STEP}"

    # --- Indexing ---
    echo "--> Indexing at thought_step=${STEP}..."

    ORIG_INDEX_YAML="$CONFIG_DIR/index.yaml"
    TEMP_INDEX_YAML="$CONFIG_DIR/index_truncation_step${STEP}_tmp.yaml"
    cp "$ORIG_INDEX_YAML" "$TEMP_INDEX_YAML"
    TEMP_FILES+=("$TEMP_INDEX_YAML")

    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TEMP_INDEX_YAML" \
        --enable_instruct True

    python mbeir_retriever.py \
        --config_path "$TEMP_INDEX_YAML" \
        --uniir_dir "$STEP_WORK_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --enable_create_index \
        --num_thought_tokens $NUM_THOUGHT_TOKENS

    # --- Retrieval ---
    echo "--> Retrieval at thought_step=${STEP}..."

    ORIG_RETRIEVAL_YAML="$CONFIG_DIR/retrieval.yaml"
    TEMP_RETRIEVAL_YAML="$CONFIG_DIR/retrieval_truncation_step${STEP}_tmp.yaml"
    cp "$ORIG_RETRIEVAL_YAML" "$TEMP_RETRIEVAL_YAML"
    TEMP_FILES+=("$TEMP_RETRIEVAL_YAML")

    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TEMP_RETRIEVAL_YAML" \
        --enable_instruct True

    STEP_LOG="$RESULTS_SUMMARY_DIR/retrieval_log_step_${STEP}.txt"

    python mbeir_retriever.py \
        --config_path "$TEMP_RETRIEVAL_YAML" \
        --uniir_dir "$STEP_WORK_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --enable_retrieval \
        --num_thought_tokens $NUM_THOUGHT_TOKENS 2>&1 | tee "$STEP_LOG"

    # 提取 Recall 指标
    echo "----------------------------------------" >> $LOG_FILE
    echo "Results for thought_step=${STEP} / ${NUM_THOUGHT_TOKENS}:" >> $LOG_FILE
    grep -E "Mean Recall@|Recall@" "$STEP_LOG" >> $LOG_FILE || echo "No Recall metrics found" >> $LOG_FILE
    echo "" >> $LOG_FILE

    echo "Finished thought_step=${STEP}"
done

# ============================================================================
# 最终汇总
# ============================================================================
echo ""
echo "############################################################################"
echo "Thought Token Truncation Evaluation Completed!"
echo "############################################################################"
echo "Checkpoint: $CKPT_NAME ($CKPT_PATH)"
echo "Evaluated ${NUM_THOUGHT_TOKENS} thought steps"
echo ""
echo "Summary Log: $LOG_FILE"
echo ""
echo "Per-step directories:"
for STEP in $(seq 1 $NUM_THOUGHT_TOKENS); do
    echo "  - step ${STEP} -> $BASE_WORK_DIR/step_${STEP}/"
done
echo "############################################################################"
echo ""
echo "=== Summary ==="
cat $LOG_FILE
