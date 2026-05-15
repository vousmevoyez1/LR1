import os
import json
import torch

# from modeling_jina_embeddings_v4 import JinaEmbeddingsV4Processor
from transformers import AutoProcessor


JSONL_PATH = "/data/M-BEIR/sub_MBEIR/cand_pool/local/mbeir_infoseek_task6_cand_pool_original.jsonl"
MODEL_PATH = "/data/jina-v4-local-copy"   # 改成你的模型路径

TOP_K = 50
BATCH_SIZE = 16

MAX_TOK_LEN = 1000


@torch.inference_mode()
def main_check():
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )

    results = []  # (token_len, did, line_id, txt)

    texts = []
    metas = []  # (did, line_id, txt)

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, 1):
            obj = json.loads(line)
            txt = obj.get("txt")
            if not isinstance(txt, str):
                print(f"Warning: line {line_id} has no valid 'txt' field, skipping...")
                continue

            texts.append(txt)
            metas.append((obj.get("did", ""), line_id, txt))

            # 到一个 batch 就跑一次 tokenizer
            if len(texts) == BATCH_SIZE:
                batch = processor.process_texts(texts)
                lengths = batch["attention_mask"].sum(dim=1).tolist()

                for l, (did, lid, t) in zip(lengths, metas):
                    results.append((int(l), did, lid, t))

                texts = []
                metas = []

    # 处理最后不足一个 batch 的部分
    if texts:
        batch = processor.process_texts(texts)
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        for l, (did, lid, t) in zip(lengths, metas):
            results.append((int(l), did, lid, t))

    # 按 token 长度排序，取前 10
    results.sort(key=lambda x: x[0], reverse=True)
    top10 = results[:TOP_K]

    print(f"\nTop-{TOP_K} longest txt by token length:\n")
    for i, (tok_len, did, lid, txt) in enumerate(top10, 1):
        print(f"{i:02d}. token_len = {tok_len}")
        print(f"    line_id = {lid}, did = {did}")
        print(f"    preview = {txt[:120].replace(chr(10), ' ')}")
        print("-" * 80)


@torch.inference_mode()
def main_change(in_path, out_path, max_tok_len=MAX_TOK_LEN):
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    kept, removed, total = 0, 0, 0

    texts = []
    objs = []

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            txt = obj.get("txt")
            if not isinstance(txt, str):
                continue

            total += 1
            texts.append(txt)
            objs.append(obj)

            if len(texts) == BATCH_SIZE:
                batch = processor.process_texts(texts, padding="longest")
                lengths = batch["attention_mask"].sum(dim=1).tolist()

                for l, o in zip(lengths, objs):
                    if int(l) <= max_tok_len:
                        fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                        kept += 1
                    else:
                        removed += 1

                texts, objs = [], []

        # 最后不足一个 batch
        if texts:
            batch = processor.process_texts(texts, padding="longest")
            lengths = batch["attention_mask"].sum(dim=1).tolist()

            for l, o in zip(lengths, objs):
                if int(l) <= max_tok_len:
                    fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    removed += 1

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Total={total}, Kept={kept}, Removed={removed}, Threshold={max_tok_len}")


if __name__ == "__main__":
    path_1 = "/data/M-BEIR/sub_MBEIR/cand_pool/local/mbeir_oven_task6_cand_pool_original.jsonl"
    path_2 = "/data/M-BEIR/sub_MBEIR/cand_pool/local/mbeir_infoseek_task6_cand_pool_original.jsonl"

    path_1_out = "/data/M-BEIR/sub_MBEIR/cand_pool/local/mbeir_oven_task6_cand_pool.jsonl"
    path_2_out = "/data/M-BEIR/sub_MBEIR/cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl"
    main_change(path_1, path_1_out, max_tok_len=MAX_TOK_LEN)
    print()
    print("#############")
    print()
    main_change(path_2, path_2_out, max_tok_len=MAX_TOK_LEN)