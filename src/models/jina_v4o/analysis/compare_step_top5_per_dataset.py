#!/usr/bin/env python3
"""
对 selected cases 中每个数据集选多个 query（默认 5 个），对比 step_1 ~ step_5 的 top5 检索结果。
输出 HTML（含文本+图片）和 JSON。
"""

import argparse
import base64
import html
import json
import mimetypes
import os
from collections import defaultdict
from typing import Any, Dict, List, Set


DEFAULT_CASES_JSON = "/data/LR1/src/models/jina_v4o/analysis/output/selected_cases_all_union_mt5_recall_mt0_miss.json"
DEFAULT_SUB_MBEIR = "/data/M-BEIR/sub_MBEIR"
DEFAULT_MBEIR_ROOT = "/data/M-BEIR"
DEFAULT_STEPS_ROOT = "/data/LR1/runs/eval_thought_truncation/MTtrunO5"
DEFAULT_OUTPUT_HTML = "/data/LR1/src/models/jina_v4o/analysis/output/compare_steps_top5_one_query_per_dataset3.html"
DEFAULT_OUTPUT_JSON = "/data/LR1/src/models/jina_v4o/analysis/output/compare_steps_top5_one_query_per_dataset3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare step-wise top5 retrieval results for multiple queries per dataset.")
    parser.add_argument("--cases_json", type=str, default=DEFAULT_CASES_JSON)
    parser.add_argument("--steps_root", type=str, default=DEFAULT_STEPS_ROOT)
    parser.add_argument("--sub_mbeir_dir", type=str, default=DEFAULT_SUB_MBEIR)
    parser.add_argument("--mbeir_root", type=str, default=DEFAULT_MBEIR_ROOT)
    parser.add_argument("--output_html", type=str, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cases_per_dataset", type=int, default=20)
    parser.add_argument("--image_width", type=int, default=170)
    parser.add_argument("--max_text_chars", type=int, default=220)
    return parser.parse_args()


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_cases_per_dataset(cases_payload: Dict[str, Any], cases_per_dataset: int) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases_payload.get("cases", []):
        ds = case.get("dataset")
        if not ds:
            continue
        if ds not in selected:
            selected[ds] = []
        if len(selected[ds]) < max(1, cases_per_dataset):
            selected[ds].append(case)
    return selected


def get_k_value(dataset: str) -> str:
    return "k50" if "fashion200k" in dataset else "k10"


def build_run_filename(dataset: str, pool: str) -> str:
    return f"mbeir_{dataset}_{pool}_pool_test_{get_k_value(dataset)}_run.txt"


def get_step_run_paths(steps_root: str, dataset: str, pool: str) -> Dict[str, str]:
    run_name = build_run_filename(dataset, pool)
    paths = {}
    for step in range(1, 6):
        p = os.path.join(
            steps_root,
            f"step_{step}",
            "retrieval_results/JinaV4/Large/Instruct/InBatch/reason_steps_5/run_files",
            run_name,
        )
        paths[f"step_{step}"] = p
    return paths


def parse_topk_for_qids(run_file: str, target_qids: Set[str], topk: int = 5) -> Dict[str, List[str]]:
    """
    TREC run line format: qid Q0 did rank score run_name task_id
    仅提取 target_qids 的 topk did。
    """
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not os.path.exists(run_file):
        return {qid: [] for qid in target_qids}

    with open(run_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            qid = parts[0]
            if qid not in target_qids:
                continue
            did = parts[2]
            try:
                rank = int(parts[3])
            except ValueError:
                continue
            if rank <= topk:
                out[qid].append({"did": did, "rank": rank})

    final: Dict[str, List[str]] = {}
    for qid in target_qids:
        arr = sorted(out.get(qid, []), key=lambda x: x["rank"])[:topk]
        final[qid] = [x["did"] for x in arr]
    return final


def list_candidate_pool_files(sub_mbeir_dir: str) -> List[str]:
    files: List[str] = []
    local_dir = os.path.join(sub_mbeir_dir, "cand_pool", "local")
    global_dir = os.path.join(sub_mbeir_dir, "cand_pool", "global")

    if os.path.isdir(local_dir):
        for name in sorted(os.listdir(local_dir)):
            if name.endswith(".jsonl"):
                files.append(os.path.join(local_dir, name))

    preferred_global = ["mbeir_union_val_cand_pool.jsonl", "mbeir_union_train_cand_pool.jsonl"]
    if os.path.isdir(global_dir):
        existing = {n for n in os.listdir(global_dir) if n.endswith(".jsonl")}
        for name in preferred_global:
            if name in existing:
                files.append(os.path.join(global_dir, name))
        for name in sorted(existing):
            if name not in preferred_global:
                files.append(os.path.join(global_dir, name))

    return files


def build_did_lookup(needed_dids: Set[str], cand_pool_files: List[str]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    remaining = set(needed_dids)

    for fp in cand_pool_files:
        if not remaining:
            break
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not remaining:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                did = obj.get("did")
                if did in remaining:
                    lookup[did] = {
                        "did": did,
                        "txt": obj.get("txt"),
                        "img_path": obj.get("img_path"),
                        "modality": obj.get("modality"),
                        "src_content": obj.get("src_content"),
                    }
                    remaining.remove(did)

    return lookup


def resolve_dids(dids: List[str], did_lookup: Dict[str, Dict[str, Any]], mbeir_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for did in dids:
        item = did_lookup.get(did)
        if not item:
            rows.append(
                {
                    "did": did,
                    "found": False,
                    "txt": None,
                    "img_path": None,
                    "img_abs_path": None,
                    "modality": None,
                }
            )
            continue

        rel = item.get("img_path")
        abs_p = os.path.join(mbeir_root, rel) if rel else None
        rows.append(
            {
                "did": did,
                "found": True,
                "txt": item.get("txt"),
                "img_path": rel,
                "img_abs_path": abs_p,
                "modality": item.get("modality"),
            }
        )
    return rows


def annotate_relevance(entries: List[Dict[str, Any]], relevant_did_set: Set[str]) -> List[Dict[str, Any]]:
    for entry in entries:
        did = entry.get("did")
        entry["is_relevant"] = did in relevant_did_set
    return entries


def _image_to_data_uri(path: str) -> str:
    if not path or (not os.path.exists(path)):
        return ""
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _truncate_text_html(text: str, max_chars: int) -> str:
    if not isinstance(text, str) or not text.strip():
        return '<span class="empty">(无文本)</span>'
    if max_chars <= 0 or len(text) <= max_chars:
        return html.escape(text)
    short = html.escape(text[:max_chars].rstrip())
    full = html.escape(text)
    return (
        f"{short}..."
        '<details class="more"><summary>展开</summary>'
        f'<div class="full-text">{full}</div>'
        "</details>"
    )


def _render_item_cell(entry: Dict[str, Any], image_width: int, max_text_chars: int, relevant_did_set: Set[str] = None) -> str:
    did = html.escape(str(entry.get("did", "")))
    relevant_did_set = relevant_did_set or set()
    is_relevant = bool(entry.get("is_relevant", entry.get("did") in relevant_did_set))

    cell_cls = "item-cell relevant-hit" if is_relevant else "item-cell"
    did_cls = "did relevant-hit-text" if is_relevant else "did"
    hit_tag = "<div class='hit-tag'>✓ relevant</div>" if is_relevant else ""

    if not entry.get("found"):
        return (
            f"<td class='{cell_cls}'>"
            f"<div class='{did_cls}'>{did}</div>"
            f"{hit_tag}"
            "<div class='missing'>未找到 did（仅高亮 did 匹配）</div>"
            "</td>"
        )

    txt_html = _truncate_text_html(entry.get("txt") or "", max_text_chars)
    modality = html.escape(str(entry.get("modality") or ""))
    rel_img = entry.get("img_path")
    abs_img = entry.get("img_abs_path")

    img_html = "<div class='empty'>(无图片)</div>"
    if rel_img and abs_img and os.path.exists(abs_img):
        data_uri = _image_to_data_uri(abs_img)
        img_html = (
            f'<img src="{data_uri}" width="{image_width}" loading="lazy" />'
            f'<div class="img-path">{html.escape(rel_img)}</div>'
        )
    elif rel_img:
        img_html = f"<div class='empty'>(路径存在但图片文件不存在)</div><div class='img-path'>{html.escape(rel_img)}</div>"

    return (
        f"<td class='{cell_cls}'>"
        f"<div class='{did_cls}'>{did}</div>"
        f"{hit_tag}"
        f"<div class='modality'>modality: {modality or '-'}</div>"
        f"<div class='txt'>{txt_html}</div>"
        f"<div class='img'>{img_html}</div>"
        "</td>"
    )


def _render_query_block(ds_item: Dict[str, Any], mbeir_root: str, image_width: int, max_text_chars: int) -> str:
    qid = html.escape(ds_item.get("qid") or "")
    qtxt_html = _truncate_text_html(ds_item.get("query_txt") or "", max_text_chars)
    qimg_rel = ds_item.get("query_img_path") or ""

    img_html = "<div class='empty'>(无图片)</div>"
    if qimg_rel:
        qimg_abs = os.path.join(mbeir_root, qimg_rel)
        if os.path.exists(qimg_abs):
            data_uri = _image_to_data_uri(qimg_abs)
            img_html = (
                f'<img src="{data_uri}" width="{image_width}" loading="lazy" />'
                f'<div class="img-path">{html.escape(qimg_rel)}</div>'
            )
        else:
            img_html = f"<div class='empty'>(路径存在但图片文件不存在)</div><div class='img-path'>{html.escape(qimg_rel)}</div>"

    return (
        "<div class='query-box'>"
        f"<div class='query-title'>qid: {qid}</div>"
        f"<div><b>query_txt:</b> <span class='query-txt'>{qtxt_html}</span></div>"
        f"<div><b>query_img:</b> <div class='query-img'>{img_html}</div></div>"
        "</div>"
    )


def render_html(report: Dict[str, Any], output_html: str, mbeir_root: str, image_width: int, max_text_chars: int) -> None:
    datasets = report.get("datasets", [])

    style = """
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { margin-bottom: 6px; }
    .meta { color: #666; margin-bottom: 18px; }
    .dataset-block { margin-bottom: 48px; }
    .query-box { border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; margin: 10px 0 16px; background: #fafafa; }
    .query-title { font-weight: bold; margin-bottom: 6px; }
    .query-txt { white-space: pre-wrap; }
    .query-img { margin-top: 6px; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    th, td { border: 1px solid #ccc; vertical-align: top; padding: 8px; }
    th.row-head { width: 140px; background: #f3f3f3; }
    .did { font-weight: bold; color: #2c3e50; margin-bottom: 4px; word-break: break-all; }
    .did.relevant-hit-text { color: #127a1f; }
    .item-cell.relevant-hit { background: #f0fff2; border-color: #7ed18a; }
    .hit-tag { display: inline-block; font-size: 12px; color: #127a1f; background: #dbf7df; border: 1px solid #7ed18a; border-radius: 10px; padding: 1px 8px; margin-bottom: 6px; }
    .modality { color: #777; font-size: 12px; margin-bottom: 6px; }
    .txt { white-space: pre-wrap; font-size: 13px; line-height: 1.4; margin-bottom: 8px; }
    .img img { border: 1px solid #ddd; border-radius: 4px; }
    .img-path { color: #777; font-size: 12px; margin-top: 4px; word-break: break-all; }
    .empty { color: #999; font-style: italic; }
    .missing { color: #b23b3b; font-weight: bold; }
    .hint { color: #666; margin-bottom: 16px; }
    details.more { display: inline; margin-left: 4px; }
    details.more summary { display: inline; color: #2563eb; cursor: pointer; }
    .full-text { white-space: pre-wrap; margin-top: 4px; }
    """

    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>Compare Top5 Across Steps</title>")
    parts.append(f"<style>{style}</style></head><body>")
    parts.append("<h1>同一 Query 在 Step1~Step5 的 Top5 检索结果对比</h1>")
    parts.append(
        f"<div class='meta'>Datasets: {len(datasets)} | "
        f"cases_per_dataset: {report.get('cases_per_dataset')} | topk: {report.get('topk')}</div>"
    )
    parts.append("<div class='hint'>每个数据集展示多个 query（默认 5 个）。每个 query 一张表；行是 relevant_dids/step_1..step_5，列是 top1..top5。</div>")

    for ds in datasets:
        ds_name = html.escape(ds.get("dataset") or "unknown")
        parts.append(f"<div class='dataset-block'><h2>{ds_name}</h2>")
        for case_item in ds.get("cases", []):
            case_idx = case_item.get("case_index_in_dataset", "?")
            case_qid = html.escape(case_item.get("qid") or "")
            parts.append(f"<h3>case #{case_idx} | qid: {case_qid}</h3>")
            parts.append(_render_query_block(case_item, mbeir_root=mbeir_root, image_width=image_width, max_text_chars=max_text_chars))

            parts.append("<table>")
            rows = [
                ("relevant_dids", case_item.get("relevant_dids_resolved", [])),
                ("step_1", case_item.get("step_top5_resolved", {}).get("step_1", [])),
                ("step_2", case_item.get("step_top5_resolved", {}).get("step_2", [])),
                ("step_3", case_item.get("step_top5_resolved", {}).get("step_3", [])),
                ("step_4", case_item.get("step_top5_resolved", {}).get("step_4", [])),
                ("step_5", case_item.get("step_top5_resolved", {}).get("step_5", [])),
            ]
            relevant_did_set = set(case_item.get("relevant_dids", []))
            for row_name, entries in rows:
                parts.append(f"<tr><th class='row-head'>{html.escape(row_name)}</th>")
                for e in entries:
                    parts.append(
                        _render_item_cell(
                            e,
                            image_width=image_width,
                            max_text_chars=max_text_chars,
                            relevant_did_set=relevant_did_set,
                        )
                    )
                parts.append("</tr>")
            parts.append("</table><br/>")
        parts.append("</div>")

    parts.append("</body></html>")

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    args = parse_args()

    payload = read_json(args.cases_json)
    selected = choose_cases_per_dataset(payload, cases_per_dataset=args.cases_per_dataset)
    pool = payload.get("metadata", {}).get("pool", "union")

    # 1) 读取每个 step 的 top5
    step_top5_by_dataset: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for dataset, cases in selected.items():
        qids = {c.get("qid") for c in cases if c.get("qid")}
        step_paths = get_step_run_paths(args.steps_root, dataset=dataset, pool=pool)
        step_top5: Dict[str, Dict[str, List[str]]] = {}
        for step_name, run_path in step_paths.items():
            top5_map = parse_topk_for_qids(run_path, target_qids=qids, topk=args.topk)
            step_top5[step_name] = top5_map
        step_top5_by_dataset[dataset] = step_top5

    # 2) 收集所有需要解析的 did
    needed_dids: Set[str] = set()
    for dataset, cases in selected.items():
        for case in cases:
            qid = case.get("qid")
            for did in case.get("relevant_dids", []) or []:
                if did:
                    needed_dids.add(did)
            for step_name in ["step_1", "step_2", "step_3", "step_4", "step_5"]:
                for did in step_top5_by_dataset.get(dataset, {}).get(step_name, {}).get(qid, []):
                    if did:
                        needed_dids.add(did)

    # 3) did -> 文本/图片 反查
    cand_pool_files = list_candidate_pool_files(args.sub_mbeir_dir)
    did_lookup = build_did_lookup(needed_dids, cand_pool_files)

    # 4) 组装结果
    datasets_out: List[Dict[str, Any]] = []
    selected_case_count = 0
    for dataset in sorted(selected.keys()):
        cases_out: List[Dict[str, Any]] = []
        for idx, case in enumerate(selected[dataset], start=1):
            qid = case.get("qid")
            case_out = {
                "case_index_in_dataset": idx,
                "qid": qid,
                "query_txt": case.get("query_txt"),
                "query_img_path": case.get("query_img_path"),
                "query_modality": case.get("query_modality"),
                "relevant_dids": case.get("relevant_dids", []),
                "step_top5": {
                    step_name: step_top5_by_dataset.get(dataset, {}).get(step_name, {}).get(qid, [])
                    for step_name in ["step_1", "step_2", "step_3", "step_4", "step_5"]
                },
            }
            relevant_did_set = set(case_out["relevant_dids"])
            case_out["relevant_dids_resolved"] = annotate_relevance(
                resolve_dids(case_out["relevant_dids"], did_lookup, args.mbeir_root),
                relevant_did_set,
            )
            case_out["step_top5_resolved"] = {}
            for step_name in ["step_1", "step_2", "step_3", "step_4", "step_5"]:
                case_out["step_top5_resolved"][step_name] = annotate_relevance(
                    resolve_dids(case_out["step_top5"].get(step_name, []), did_lookup, args.mbeir_root),
                    relevant_did_set,
                )
            cases_out.append(case_out)
            selected_case_count += 1

        datasets_out.append({"dataset": dataset, "cases": cases_out})

    unresolved = sorted([d for d in needed_dids if d not in did_lookup])

    report = {
        "input_cases_json": args.cases_json,
        "pool": pool,
        "steps_root": args.steps_root,
        "topk": args.topk,
        "cases_per_dataset": args.cases_per_dataset,
        "selected_dataset_count": len(datasets_out),
        "selected_case_count": selected_case_count,
        "selected_datasets": [x["dataset"] for x in datasets_out],
        "needed_did_count": len(needed_dids),
        "resolved_did_count": len(needed_dids) - len(unresolved),
        "unresolved_dids": unresolved,
        "cand_pool_files": cand_pool_files,
        "datasets": datasets_out,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    render_html(
        report,
        output_html=args.output_html,
        mbeir_root=args.mbeir_root,
        image_width=args.image_width,
        max_text_chars=args.max_text_chars,
    )

    print("Done.")
    print(f"Datasets: {len(datasets_out)}")
    print(f"Cases: {selected_case_count} (target per dataset={args.cases_per_dataset})")
    print(f"Needed DIDs: {len(needed_dids)}, resolved: {len(needed_dids) - len(unresolved)}, unresolved: {len(unresolved)}")
    print(f"JSON: {args.output_json}")
    print(f"HTML: {args.output_html}")


if __name__ == "__main__":
    main()
