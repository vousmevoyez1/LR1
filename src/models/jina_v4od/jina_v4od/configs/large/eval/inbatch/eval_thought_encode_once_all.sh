#!/bin/bash
set -e

# 一次编码 + 一次评估，完成三种指标：
# 1) Max-Sim
# 2) Oracle Best-Step
# 3) Last-Step

SRC="/data/LR1/src"
COMMON_DIR="$SRC/common"
export PYTHONPATH="$SRC"

# ====== 需按实验修改 ======
UNIIR_DIR="/data/LR1/runs/eval_thought_once/MtestT5/JinaV4/Large/Instruct/InBatch"
MBEIR_DATA_DIR="/data/M-BEIR/sub_MBEIR"
EMBED_YAML="/data/LR1/src/models/jina_v4od/jina_v4od/configs/large/eval/inbatch/embed.yaml"
RETRIEVAL_YAML="/data/LR1/src/models/jina_v4od/jina_v4od/configs/large/eval/inbatch/retrieval.yaml"
NUM_THOUGHT_TOKENS=5
SPLIT="test"

# 候选池复用：不再编码 candidate
CAND_POOL_DIR="/data/LR1/runs/eval_finetuned/MtestT5/embed/JinaV4/Large/Instruct/InBatch/reason_steps_5/cand_pool"
CAND_INDEX_DIR="/data/LR1/runs/eval_finetuned/MtestT5/indices/JinaV4/Large/Instruct/InBatch/reason_steps_5/cand_pool"

# 分布式编码参数
CUDA_VISIBLE_DEVICES="0,1"
NPROC=2
MASTER_PORT=29901
# 可选：LoRA ckpt，不需要可留空
LORA_CHECKPOINT="/data/LR1/checkpointMTO5/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth"
# ==========================

export CUDA_VISIBLE_DEVICES
export MASTER_ADDR=127.0.0.1
export MASTER_PORT

mkdir -p "$UNIIR_DIR"

cd "$COMMON_DIR"

TMP_EMBED_YAML="/data/LR1/src/models/jina_v4od/jina_v4od/configs/large/eval/inbatch/embed_encode_once_tmp.yaml"
cp "$EMBED_YAML" "$TMP_EMBED_YAML"
trap 'rm -f "$TMP_EMBED_YAML"' EXIT

# 仅编码 query；candidate 走复用目录
python - <<'PY'
import yaml
p = "/data/LR1/src/models/jina_v4od/jina_v4od/configs/large/eval/inbatch/embed_encode_once_tmp.yaml"
with open(p, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["embed_config"]["cand_pools_config"]["enable_embed"] = False
cfg["embed_config"]["test_datasets_config"]["enable_embed"] = True

with open(p, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

python config_updater.py \
  --update_mbeir_yaml_instruct_status \
  --mbeir_yaml_file_path "$TMP_EMBED_YAML" \
  --enable_instruct True

CMD=(
  python -m torch.distributed.run
  --nproc_per_node="$NPROC"
  --master_port="$MASTER_PORT"
  --master_addr="$MASTER_ADDR"
  /data/LR1/src/common/mbeir_embedder_per_step.py
  --config_path "$TMP_EMBED_YAML"
  --uniir_dir "$UNIIR_DIR"
  --mbeir_data_dir "$MBEIR_DATA_DIR"
  --num_thought_tokens "$NUM_THOUGHT_TOKENS"
  --all_steps
)

if [ -n "$LORA_CHECKPOINT" ]; then
  CMD+=(--lora_checkpoint "$LORA_CHECKPOINT")
fi

"${CMD[@]}"

# 一次运行评估脚本，产出三种结果
python /data/LR1/src/models/jina_v4od/analysis/eval_thought_retrieval.py \
  --uniir_dir "$UNIIR_DIR" \
  --mbeir_data_dir "$MBEIR_DATA_DIR" \
  --retrieval_config "$RETRIEVAL_YAML" \
  --num_thought_tokens "$NUM_THOUGHT_TOKENS" \
  --mode "all" \
  --split "$SPLIT" \
  --cand_pool_dir "$CAND_POOL_DIR" \
  --cand_index_dir "$CAND_INDEX_DIR" \
  --query_batch_size 1024
