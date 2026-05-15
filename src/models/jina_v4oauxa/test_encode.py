#!/usr/bin/env python3
"""
Simple test script for Jina V4 encoding.

Supports:
1) Text encoding
2) Image encoding
3) Image + prompt text (multimodal) encoding

Note:
Current model code no longer supports text `prefix` in `process_texts`.
So this script uses `processor.process_texts/process_images/process_multimodal`
plus direct `model(...)` forward for compatibility.

Examples:
  # Encode text
  python /data/LR1/src/models/jina_v4o/test_encode.py \
      --model-path /data/jina-v4-local-copy \
      --text "a cat sitting on a sofa" \
      --task retrieval

  # Encode image
  python /data/LR1/src/models/jina_v4o/test_encode.py \
      --model-path /data/jina-v4-local-copy \
      --image /path/to/image.jpg \
      --task retrieval

  # Encode image with prompt text (multimodal)
  python /data/LR1/src/models/jina_v4o/test_encode.py \
      --model-path /data/jina-v4-local-copy \
      --image /path/to/image.jpg \
      --prompt "Find relevant products in this image" \
      --task retrieval

  # Batch encode from txt (one text per line) and print pairwise similarity
  python /data/LR1/src/models/jina_v4o/test_encode.py \
      --model-path /data/jina-v4-local-copy \
      --input-txt /path/to/samples.txt \
      --save-sim-csv /path/to/similarity.csv

  # Batch encode from jsonl (each line: {"id","text","image","prompt"})
  python /data/LR1/src/models/jina_v4o/test_encode.py \
      --model-path /data/jina-v4-local-copy \
      --input-jsonl /path/to/samples.jsonl \
      --save-sim-csv /path/to/similarity.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


# Ensure /data/LR1/src is importable when running this file directly
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.jina_v4o.jina_v4o.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Jina V4 text/image encoding")

    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/jina-v4-local-copy",
        help="Base pretrained model path (local dir or HF repo).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/data/LR1/checkpointMTO0/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth",
        help="Optional finetuned LoRA checkpoint (.pth) to load.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="retrieval",
        help="Task adapter name, e.g. retrieval / text-matching / code.",
    )

    input_group = parser.add_argument_group("input")
    input_group.add_argument("--text", type=str, default="", help="Text to encode.")
    input_group.add_argument("--image", type=str, default="", help="Image path to encode.")
    input_group.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Extra prompt/instruction text. If --image is set, this triggers multimodal encoding.",
    )
    input_group.add_argument(
        "--prompt-name",
        type=str,
        default="query",
        choices=["query", "passage"],
        help="Reserved for compatibility; currently not used by this script.",
    )

    parser.add_argument("--num-thought-tokens", type=int, default=0)
    parser.add_argument(
        "--thought-pooling-mode",
        type=str,
        default="last",
        choices=["last", "mean_all_thought_tokens"],
    )
    parser.add_argument("--truncate-dim", type=int, default=0, help="0 means disabled.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"])

    parser.add_argument("--save-npy", type=str, default="", help="Optional output .npy path.")
    parser.add_argument("--print-dim", type=int, default=16, help="Print first N dimensions.")
    parser.add_argument(
        "--quiet-input",
        action="store_true",
        help="Disable verbose input dump (enabled by default).",
    )
    parser.add_argument(
        "--quiet-embedding",
        action="store_true",
        help="Disable full embedding vector print (enabled by default).",
    )

    # Batch mode
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="",
        help=(
            "Batch input file (.jsonl). Each line should be a JSON object with optional keys: "
            "id, text, image, prompt."
        ),
    )
    parser.add_argument(
        "--input-txt",
        type=str,
        default="",
        help="Batch text file (.txt), one text sample per line.",
    )
    parser.add_argument(
        "--save-sim-csv",
        type=str,
        default="",
        help="Optional path to save pairwise similarity matrix as CSV.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(args: argparse.Namespace, device: torch.device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[INFO] Loading base model from: {args.model_path}")
    model = JinaEmbeddingsV4Model.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
    )

    model.task = args.task
    model.thought_pooling_mode = args.thought_pooling_mode

    if args.num_thought_tokens > 0:
        print(f"[INFO] Setup thought tokens: {args.num_thought_tokens}")
        model.setup_thought_tokens(args.num_thought_tokens, skip_init=True)

    if args.checkpoint_path:
        ckpt = Path(args.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"[INFO] Loading finetuned checkpoint: {ckpt}")
        from models.jina_v4o.train import load_lora_checkpoint

        model, _ = load_lora_checkpoint(model, str(ckpt))

    model.eval()
    model = model.to(device)
    return model


def _forward_single_embedding(model, processed: Dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    device = next(model.parameters()).device
    processed = {k: v.to(device) for k, v in processed.items()}

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(
                **processed,
                task_label=args.task,
                num_thought_tokens=args.num_thought_tokens,
                thought_pooling_mode=args.thought_pooling_mode,
            )
            emb = out.single_vec_emb[0]

    if args.truncate_dim > 0:
        emb = emb[: args.truncate_dim]
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

    return emb.detach().cpu().float()


def _decode_input_ids_if_possible(model, input_ids: torch.Tensor) -> Optional[str]:
    tokenizer = None

    if hasattr(model, "processor") and hasattr(model.processor, "tokenizer"):
        tokenizer = model.processor.tokenizer
    elif hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer

    if tokenizer is None:
        return None

    try:
        ids = input_ids[0].detach().cpu().tolist()
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        return None


def dump_final_input(
    model,
    args: argparse.Namespace,
    *,
    mode: str,
    text: str = "",
    prompt: str = "",
    image_path: str = "",
    processed: Optional[Dict[str, torch.Tensor]] = None,
    sample_id: str = "",
):
    if args.quiet_input:
        return

    title = f"[DEBUG][{mode}]"
    if sample_id:
        title = f"[DEBUG][{mode}][{sample_id}]"

    # print("-" * 80)
    # print(f"{title} Final Assembled Input")
    # print(f"{title} prompt={json.dumps(prompt, ensure_ascii=False)}")
    # print(f"{title} text={json.dumps(text, ensure_ascii=False)}")
    # print(f"{title} image_path={json.dumps(image_path, ensure_ascii=False)}")

    if processed is None:
        print("-" * 80)
        return

    print(f"{title} processed_keys={list(processed.keys())}")
    for key, value in processed.items():
        if isinstance(value, torch.Tensor):
            print(
                f"{title} {key}.shape={tuple(value.shape)} "
                f"dtype={value.dtype} device={value.device}"
            )
            if key in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}:
                print(f"{title} {key}.values={value[0].detach().cpu().tolist()}")
        else:
            print(f"{title} {key}={value}")

    if isinstance(processed.get("input_ids", None), torch.Tensor):
        decoded = _decode_input_ids_if_possible(model, processed["input_ids"])
        if decoded is not None:
            print(f"{title} decoded_input={json.dumps(decoded, ensure_ascii=False)}")

    print("-" * 80)


def dump_embedding(args: argparse.Namespace, emb: torch.Tensor, mode: str, sample_id: str = ""):
    title = f"[DEBUG][{mode}]"
    if sample_id:
        title = f"[DEBUG][{mode}][{sample_id}]"

    # print(f"{title} embedding_shape={tuple(emb.shape)}")
    # print(f"{title} embedding_norm={torch.norm(emb, p=2).item():.6f}")
    # print(f"{title} embedding_first_{min(args.print_dim, emb.shape[0])}_dims={json.dumps(emb[:args.print_dim].tolist(), ensure_ascii=False)}")

    # if not args.quiet_embedding:
    #     print(f"{title} embedding_full={json.dumps(emb.tolist(), ensure_ascii=False)}")


def encode_text(model, text: str, prompt: str, args: argparse.Namespace):
    final_text = text
    if prompt.strip():
        final_text = f"{prompt.strip()}\n{text}"

    processed = model.processor.process_texts(
        texts=[final_text],
        num_thought_tokens=args.num_thought_tokens,
    )

    dump_final_input(
        model,
        args,
        mode="text",
        text=final_text,
        prompt=prompt,
        image_path="",
        processed=processed,
    )
    return _forward_single_embedding(model, processed, args)


def encode_image_only(model, image_path: str, args: argparse.Namespace):
    image = Image.open(image_path).convert("RGB")
    processed = model.processor.process_images(
        images=[image],
        num_thought_tokens=args.num_thought_tokens,
    )

    dump_final_input(
        model,
        args,
        mode="image",
        text="",
        prompt="",
        image_path=image_path,
        processed=processed,
    )
    return _forward_single_embedding(model, processed, args)


def encode_multimodal(model, image_path: str, prompt: str, args: argparse.Namespace):
    image = Image.open(image_path).convert("RGB")
    text = prompt.strip()
    if not text:
        raise ValueError("Multimodal mode requires non-empty --prompt.")

    processed = model.processor.process_multimodal(
        images=[image],
        texts=[text],
        num_thought_tokens=args.num_thought_tokens,
    )

    dump_final_input(
        model,
        args,
        mode="multimodal",
        text=text,
        prompt=prompt,
        image_path=image_path,
        processed=processed,
    )
    return _forward_single_embedding(model, processed, args)


def load_batch_samples(args: argparse.Namespace) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []

    if args.input_jsonl:
        p = Path(args.input_jsonl)
        if not p.exists():
            raise FileNotFoundError(f"input-jsonl not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_no} in {p}: {e}") from e

                if not isinstance(obj, dict):
                    raise ValueError(f"Line {line_no} in {p} is not a JSON object")

                sample = {
                    "id": str(obj.get("id", f"sample_{len(samples)}")),
                    "text": str(obj.get("text", "") or ""),
                    "image": str(obj.get("image", "") or ""),
                    "prompt": str(obj.get("prompt", "") or ""),
                }
                if not sample["text"] and not sample["image"]:
                    raise ValueError(
                        f"Line {line_no} in {p} has neither text nor image. "
                        "At least one of 'text' or 'image' is required."
                    )
                samples.append(sample)

    elif args.input_txt:
        p = Path(args.input_txt)
        if not p.exists():
            raise FileNotFoundError(f"input-txt not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                samples.append(
                    {
                        "id": f"sample_{idx}",
                        "text": text,
                        "image": "",
                        "prompt": "",
                    }
                )

    return samples


def encode_one_sample(model, sample: Dict[str, str], args: argparse.Namespace) -> Tuple[torch.Tensor, str]:
    text = sample.get("text", "").strip()
    image = sample.get("image", "").strip()
    prompt = sample.get("prompt", "").strip()

    if image:
        img_path = Path(image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

    if image and prompt:
        emb = encode_multimodal(model, image, prompt, args)
        mode = "multimodal"
    elif image:
        emb = encode_image_only(model, image, args)
        mode = "image"
    elif text:
        emb = encode_text(model, text, prompt, args)
        mode = "text"
    else:
        raise ValueError("Invalid sample: requires text and/or image")

    sample_id = sample.get("id", "")
    dump_embedding(args, emb, mode=mode, sample_id=sample_id)

    return emb, mode


def compute_pairwise_cosine(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    sim = normalized @ normalized.T
    return sim


def print_similarity_matrix(labels: List[str], sim: torch.Tensor):
    print("=" * 80)
    print("[RESULT] pairwise cosine similarity matrix")
    print(f"[RESULT] matrix_shape={tuple(sim.shape)}")

    header = ["id"] + labels
    print("\t".join(header))
    for i, label in enumerate(labels):
        row = [label] + [f"{sim[i, j].item():.6f}" for j in range(sim.shape[1])]
        print("\t".join(row))
    print("=" * 80)


def save_similarity_csv(labels: List[str], sim: torch.Tensor, csv_path: str):
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{sim[i, j].item():.8f}" for j in range(sim.shape[1])])

    print(f"[RESULT] saved similarity csv: {out_path}")


def main():
    args = parse_args()

    batch_mode = bool(args.input_jsonl or args.input_txt)

    if not batch_mode and not args.text and not args.image:
        raise ValueError("Please provide at least one input: --text or --image")

    if args.image and not batch_mode:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

    device = resolve_device(args.device)
    model = load_model(args, device)

    if batch_mode:
        samples = load_batch_samples(args)
        if not samples:
            raise ValueError("No valid samples found in batch input.")

        print(f"[INFO] Loaded {len(samples)} samples for batch encoding")

        labels: List[str] = []
        embs: List[torch.Tensor] = []
        for idx, sample in enumerate(samples):
            emb, mode = encode_one_sample(model, sample, args)
            sample_id = sample.get("id", f"sample_{idx}")
            labels.append(sample_id)
            embs.append(emb)
            print(f"[INFO] Encoded sample {idx + 1}/{len(samples)}: id={sample_id}, mode={mode}, dim={emb.shape[0]}")

        batch_emb = torch.stack(embs, dim=0)
        sim = compute_pairwise_cosine(batch_emb)

        print_similarity_matrix(labels, sim)

        if args.save_npy:
            out_path = Path(args.save_npy)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np

            np.save(out_path, batch_emb.numpy())
            print(f"[RESULT] saved_npy={out_path}")

        if args.save_sim_csv:
            save_similarity_csv(labels, sim, args.save_sim_csv)

        return

    if args.image and args.prompt.strip():
        mode = "multimodal(image+prompt)"
        emb = encode_multimodal(model, args.image, args.prompt, args)
    elif args.image:
        mode = "image"
        emb = encode_image_only(model, args.image, args)
    else:
        mode = "text"
        emb = encode_text(model, args.text, args.prompt, args)

    dump_embedding(args, emb, mode=mode)

    vector = emb.tolist()
    show_n = min(args.print_dim, len(vector))

    print("=" * 80)
    print(f"[RESULT] mode={mode}")
    print(f"[RESULT] shape={tuple(emb.shape)}")
    print(f"[RESULT] norm={torch.norm(emb, p=2).item():.6f}")
    print(f"[RESULT] first_{show_n}_dims={json.dumps(vector[:show_n], ensure_ascii=False)}")

    if args.save_npy:
        out_path = Path(args.save_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.save(out_path, emb.numpy())
        print(f"[RESULT] saved_npy={out_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
