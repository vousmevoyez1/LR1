#!/bin/bash
set -e

# 仅新增于 jina_v4od 目录：评估 Max-Sim / Oracle-Best-Step / Last-Step

SRC="/data/LR1/src"
export PYTHONPATH="$SRC"

# ====== 需要按你的实验实际修改 ======
UNIIR_DIR="/data/LR1/runs/sdtest/MtestT5/reason_steps_5/JinaV4/Large/Instruct/InBatch"
MBEIR_DATA_DIR="/data/M-BEIR/sub_MBEIR"
RETRIEVAL_YAML="/data/LR1/src/models/jina_v4od/jina_v4od/configs/large/eval/inbatch/retrieval.yaml"
NUM_THOUGHT_TOKENS=5
MODE="all"   # maxsim | oracle_best_step | last_step | both | all
SPLIT="test"
CAND_POOL_DIR="/data/LR1/runs/eval_finetuned/MtestT5/embed/JinaV4/Large/Instruct/InBatch/reason_steps_5/cand_pool"
CAND_INDEX_DIR="/data/LR1/runs/eval_finetuned/MtestT5/indices/JinaV4/Large/Instruct/InBatch/reason_steps_5/cand_pool"
# ====================================

python /data/LR1/src/models/jina_v4od/analysis/eval_thought_retrieval.py \
  --uniir_dir "$UNIIR_DIR" \
  --mbeir_data_dir "$MBEIR_DATA_DIR" \
  --retrieval_config "$RETRIEVAL_YAML" \
  --num_thought_tokens "$NUM_THOUGHT_TOKENS" \
  --mode "$MODE" \
  --split "$SPLIT" \
  --cand_pool_dir "$CAND_POOL_DIR" \
  --cand_index_dir "$CAND_INDEX_DIR" \
  --query_batch_size 1024
