#!/bin/bash

set -e  # 遇到错误立即退出

# ============================================================================
# Qwen3-VL 微调 Checkpoint 评估脚本
# 适配 Thought Token 机制与 5090 多卡环境
# ============================================================================

# --- 基础路径配置 ---
SRC="/data/LR1/src"
COMMON_DIR="$SRC/common"
UNIIR_BASE_DIR="/data/LR1"
MBEIR_DATA_DIR="/data/M-BEIR/sub_MBEIR/"

# 修改为 Qwen3 路径
MODEL="qwen3/qwen3"
MODEL_DIR="$SRC/models/$MODEL"
SIZE="2b"  # Qwen3-VL 规模
MODE="eval"
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs/$SIZE/$MODE/$EXP_NAME"

# --- 环境变量设置 ---
# 根据之前 nvidia-smi 的结果，使用后四张 5090 (4,5,6,7)
export CUDA_VISIBLE_DEVICES=4,5
# 禁用显卡间的 Peer-to-Peer 通信（防止跨卡分配 L2Norm 算子失败）
export CUDA_P2P_DISABLE=1
# 强制开启统一内存支持（有时能缓解计算镜像缺失）
export CUDA_MODULE_LOADING=LAZY
# 限制底层数学库的 CPU 线程数，防止多进程环境下的线程爆炸
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
NPROC=2
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# 切换工作目录
cd $COMMON_DIR

# --- 微调 Checkpoint 配置 ---
# 格式: "名称:checkpoint路径:训练时TT数:评估时TT数:评估时是否启用FinalToken(true/false)"
# 注意：Qwen3-VL 的训练通常开启了 LoRA，这里填入 .pth 路径
# FinalToken 默认为 false（可省略）
CHECKPOINTS=(
    # #"Qwen3_TT161_Base2:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt16_f1/qwen3vl_epoch_1.pth:16:16:true"
    # #"Qwen3_TT41_Base2:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt4_f1/qwen3vl_epoch_1.pth:4:4:true"
    # "Qwen3_TT81_Base1:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt8_f1/qwen3vl_epoch_4.pth:8:8:true"
    # "Qwen3_TT161_Base1:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt16_f1/qwen3vl_epoch_4.pth:16:16:true"
    # #"Qwen3_TT01_Base1:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt0_f1/qwen3vl_epoch_0.pth:0:0:true"
    # #"Qwen3_TT21_Base1:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt2_f1/qwen3vl_epoch_4.pth:2:2:true"
    # #"Qwen3_TT01_Base1:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt0_f1/qwen3vl_epoch_0.pth:0:0:true"
    # #"Qwen3_TT41_Base1:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt4_f1/qwen3vl_epoch_4.pth:4:4:true"
    # #"Qwen3_TT01_n_1:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt0_f1/qwen3vl_epoch_4.pth:0:0:true"
    # # 其他示例：
    # # "Qwen3_TT16_Base16:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt16/qwen3vl_step_50.pth:16:16:true"
    "Qwen3_TT0_FT:/data/LR1/checkpoint/Qwen3VL/2b/Instruct/InBatch_tt0_f0/qwen3vl_epoch_2.pth:0:0:false"
    # "Qwen3_TT8_Base8:/data/LR1/checkpoint/qwen3/your_ckpt.pth:8:8:false"
)

# --- Qwen3 显存安全配置 ---
# 保持与训练一致，防止评估超长样本时 OOM
MAX_TOKEN_LEN=2048
MAX_PIXELS=501760

# --- Debug 配置（仅 Step1 编码阶段会进入模型）---
DEBUG_MODE=${DEBUG_MODE:-false}
DEBUG_PRINT_FREQ=${DEBUG_PRINT_FREQ:-10}
DEBUG_MAX_TEXT_CHARS=${DEBUG_MAX_TEXT_CHARS:-160}
DEBUG_TOKEN_PREVIEW_LEN=${DEBUG_TOKEN_PREVIEW_LEN:-12}

# --- 结果汇总目录 ---
RESULTS_SUMMARY_DIR="$UNIIR_BASE_DIR/retrieval_results_summary/qwen3_eval"
mkdir -p $RESULTS_SUMMARY_DIR
LOG_FILE="$RESULTS_SUMMARY_DIR/eval_summary_$(date +%Y%m%d_%H%M%S).log"

echo "Qwen3-VL Evaluation Summary - $(date)" > $LOG_FILE
echo "============================================================================" >> $LOG_FILE

# --- 自动清理机制 ---
TEMP_FILES=()
cleanup() {
    rm -f "${TEMP_FILES[@]}"
    echo "Cleanup temporary config files."
}
trap cleanup EXIT

# ============================================================================
# 主循环
# ============================================================================
RUN_IDX=0
for CKPT_CONFIG in "${CHECKPOINTS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    IFS=':' read -r CKPT_NAME CKPT_PATH TRAIN_TT EVAL_TT EVAL_FT <<< "$CKPT_CONFIG"
    
    # FinalToken 默认为 false 如果未提供
    EVAL_FT=${EVAL_FT:-false}
    
    echo ">>>> Processing: $CKPT_NAME (Path: $CKPT_PATH)"
    echo "     Config: TT=$EVAL_TT, FinalToken=$EVAL_FT"
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "SKIP: $CKPT_PATH not found."
        continue
    fi

    # 独立工作目录
    CURRENT_UNIIR_DIR="$UNIIR_BASE_DIR/runs/qwen3_eval/$CKPT_NAME"
    mkdir -p "$CURRENT_UNIIR_DIR"
    
    export MASTER_PORT=$((29800 + RUN_IDX))
    export MASTER_ADDR=127.0.0.1

    # ------------------------------------------------------------------------
    # Step 1: Embedding (重新编码 Query 和 Candidate)
    # ------------------------------------------------------------------------
    echo "--> Step 1: Encoding with Qwen3-VL..."
    
    TEMP_EMBED_YAML="$CONFIG_DIR/embed_${CKPT_NAME}_tmp.yaml"
    cp "$CONFIG_DIR/embed.yaml" "$TEMP_EMBED_YAML"
    TEMP_FILES+=("$TEMP_EMBED_YAML")

    # 更新 YAML 中的指令状态
    python config_updater.py --update_mbeir_yaml_instruct_status --mbeir_yaml_file_path "$TEMP_EMBED_YAML" --enable_instruct True

    # 运行多卡 Embedder
    # 加入了 --max_token_length 和 --max_visual_pixels 保证安全
    EXTRA_DEBUG_ARGS=()
    if [ "$DEBUG_MODE" = "true" ]; then
        EXTRA_DEBUG_ARGS+=(--debug_mode)
        EXTRA_DEBUG_ARGS+=(--debug_print_freq "$DEBUG_PRINT_FREQ")
        EXTRA_DEBUG_ARGS+=(--debug_max_text_chars "$DEBUG_MAX_TEXT_CHARS")
        EXTRA_DEBUG_ARGS+=(--debug_token_preview_len "$DEBUG_TOKEN_PREVIEW_LEN")
        echo "[DEBUG] Eval embedder debug enabled: freq=$DEBUG_PRINT_FREQ, max_chars=$DEBUG_MAX_TEXT_CHARS, token_preview=$DEBUG_TOKEN_PREVIEW_LEN"
    fi

    torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
        mbeir_embedderqwen.py \
        --config_path "$TEMP_EMBED_YAML" \
        --uniir_dir "$CURRENT_UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --num_thought_tokens $EVAL_TT \
        --enable_final_token $EVAL_FT \
        --lora_checkpoint "$CKPT_PATH" \
        --max_token_length $MAX_TOKEN_LEN \
        --max_visual_pixels $MAX_PIXELS \
        "${EXTRA_DEBUG_ARGS[@]}"

    # ------------------------------------------------------------------------
    # Step 2: Indexing
    # ------------------------------------------------------------------------
    echo "--> Step 2: Building Index..."
    TEMP_INDEX_YAML="$CONFIG_DIR/index_${CKPT_NAME}_tmp.yaml"
    cp "$CONFIG_DIR/index.yaml" "$TEMP_INDEX_YAML"
    TEMP_FILES+=("$TEMP_INDEX_YAML")

    python mbeir_retrieverqwen.py \
        --config_path "$TEMP_INDEX_YAML" \
        --uniir_dir "$CURRENT_UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --enable_create_index \
        --num_thought_tokens $EVAL_TT \
        --enable_final_token $EVAL_FT

    # ------------------------------------------------------------------------
    # Step 3: Retrieval & Recall 计算
    # ------------------------------------------------------------------------
    echo "--> Step 3: Evaluating Retrieval..."
    TEMP_RETRIEVAL_YAML="$CONFIG_DIR/retrieval_${CKPT_NAME}_tmp.yaml"
    cp "$CONFIG_DIR/retrieval.yaml" "$TEMP_RETRIEVAL_YAML"
    TEMP_FILES+=("$TEMP_RETRIEVAL_YAML")

    STEP_LOG="$RESULTS_SUMMARY_DIR/log_${CKPT_NAME}.txt"

    python mbeir_retrieverqwen.py \
        --config_path "$TEMP_RETRIEVAL_YAML" \
        --uniir_dir "$CURRENT_UNIIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --enable_retrieval \
        --num_thought_tokens $EVAL_TT \
        --enable_final_token $EVAL_FT 2>&1 | tee "$STEP_LOG"

    # 结果提取
    echo "----------------------------------------" >> $LOG_FILE
    echo "CKPT: $CKPT_NAME | TT: $EVAL_TT | FinalToken: $EVAL_FT" >> $LOG_FILE
    grep -E "Mean Recall@|Recall@" "$STEP_LOG" >> $LOG_FILE || echo "No Metrics Found" >> $LOG_FILE
done

echo "Evaluation Finished. Summary at: $LOG_FILE"
cat $LOG_FILE