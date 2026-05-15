#!/bin/bash
set -e

# Max-Sim Thought 检索评估（仅评估，不做 candidate 编码）

SRC="/data/LR1/src"
export PYTHONPATH="$SRC"

UNIIR_DIR="/data/LR1/runs/sdtest/MtestT5/reason_steps_5/JinaV4/Large/Instruct/InBatch"
MBEIR_DATA_DIR="/data/M-BEIR/sub_MBEIR"
RETRIEVAL_YAML="/data/LR1/src/models/jina_v4od/jina_v4od/configs/large/eval/inbatch/retrieval.yaml"
NUM_THOUGHT_TOKENS=5
SPLIT="test"

# 固定复用候选向量，不再编码 candidate
CAND_POOL_DIR="/data/LR1/runs/eval_finetuned/MtestT5/embed/JinaV4/Large/Instruct/InBatch/reason_steps_5/cand_pool"
CAND_INDEX_DIR="/data/LR1/runs/eval_finetuned/MtestT5/indices/JinaV4/Large/Instruct/InBatch/reason_steps_5/cand_pool"

python /data/LR1/src/models/jina_v4od/analysis/eval_thought_retrieval.py \
  --uniir_dir "$UNIIR_DIR" \
  --mbeir_data_dir "$MBEIR_DATA_DIR" \
  --retrieval_config "$RETRIEVAL_YAML" \
  --num_thought_tokens "$NUM_THOUGHT_TOKENS" \
  --mode "maxsim" \
  --split "$SPLIT" \
  --cand_pool_dir "$CAND_POOL_DIR" \
  --cand_index_dir "$CAND_INDEX_DIR" \
  --query_batch_size 128
