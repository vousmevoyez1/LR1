#!/usr/bin/env python3
"""
从筛选 case JSON 中为每个数据集选择多个 case（默认 5 个），解析 relevant_dids / mt0_top5 / mt5_top5
对应的文本与图片，并输出可视化 HTML 表格与结构化 JSON。

默认输入：
/data/LR1/src/models/jina_v4o/analysis/output/selected_cases_all_union_mt5_recall_mt0_miss.json
"""

import argparse
import base64
import html
import json
import mimetypes
import os
from typing import Dict, List, Any, Set


DEFAULT_CASES_JSON = "/data/LR1/src/models/jina_v4o/analysis/output/selected_cases_all_union_mt5_recall_mt0_miss.json"
DEFAULT_MBEIR_ROOT = "/data/M-BEIR"
DEFAULT_SUB_MBEIR = "/data/M-BEIR/sub_MBEIR"
DEFAULT_OUTPUT_HTML = "/data/LR1/src/models/jina_v4o/analysis/output/one_case_per_dataset_tables.html"
DEFAULT_OUTPUT_JSON = "/data/LR1/src/models/jina_v4o/analysis/output/one_case_per_dataset_tables.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one-case-per-dataset retrieval tables.")
    parser.add_argument("--cases_json", type=str, default=DEFAULT_CASES_JSON, help="Path to selected cases JSON.")
    parser.add_argument("--mbeir_root", type=str, default=DEFAULT_MBEIR_ROOT, help="M-BEIR root for image absolute paths.")
    parser.add_argument("--sub_mbeir_dir", type=str, default=DEFAULT_SUB_MBEIR, help="sub_MBEIR root for cand_pool files.")
    parser.add_argument("--output_html", type=str, default=DEFAULT_OUTPUT_HTML, help="Output HTML path.")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON, help="Output JSON path.")
    parser.add_argument(
        "--cases_per_dataset",
        type=int,
        default=5,
        help="How many cases to select for each dataset.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=180,
        help="Image display width in HTML (pixels).",
    )
    parser.add_argument(
        "--max_text_chars",
        type=int,
        default=240,
        help="Maximum chars to show before truncation in HTML.",
    )
    return parser.parse_args()


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_cases_per_dataset(cases_payload: Dict[str, Any], cases_per_dataset: int) -> Dict[str, List[Dict[str, Any]]]:
    cases: List[Dict[str, Any]] = cases_payload.get("cases", [])
    selected: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        dataset = case.get("dataset")
        if not dataset:
            continue
        if dataset not in selected:
            selected[dataset] = []
        if len(selected[dataset]) < max(1, cases_per_dataset):
            selected[dataset].append(case)
    return selected


def collect_needed_dids(cases_map: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
    needed: Set[str] = set()
    for cases in cases_map.values():
        for case in cases:
            for key in ("relevant_dids", "mt0_top5", "mt5_top5"):
                for did in case.get(key, []) or []:
                    if did:
                        needed.add(did)
    return needed


def list_candidate_pool_files(sub_mbeir_dir: str) -> List[str]:
    files: List[str] = []

    local_dir = os.path.join(sub_mbeir_dir, "cand_pool", "local")
    global_dir = os.path.join(sub_mbeir_dir, "cand_pool", "global")

    if os.path.isdir(local_dir):
        for name in sorted(os.listdir(local_dir)):
            if name.endswith(".jsonl"):
                files.append(os.path.join(local_dir, name))

    # 优先 val（一般包含评测用候选），其次 train（兜底）
    preferred_global = [
        "mbeir_union_val_cand_pool.jsonl",
        "mbeir_union_train_cand_pool.jsonl",
    ]
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
    if not needed_dids:
        return lookup

    remaining = set(needed_dids)

    for fp in cand_pool_files:
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
                        "source_file": fp,
                    }
                    remaining.remove(did)

        if not remaining:
            break

    return lookup


def resolve_did_items(dids: List[str], did_lookup: Dict[str, Dict[str, Any]], mbeir_root: str) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for did in dids:
        item = did_lookup.get(did)
        if not item:
            resolved.append(
                {
                    "did": did,
                    "found": False,
                    "txt": None,
                    "img_path": None,
                    "img_abs_path": None,
                    "modality": None,
                    "src_content": None,
                }
            )
            continue

        rel_img_path = item.get("img_path")
        abs_img_path = os.path.join(mbeir_root, rel_img_path) if rel_img_path else None

        resolved.append(
            {
                "did": did,
                "found": True,
                "txt": item.get("txt"),
                "img_path": rel_img_path,
                "img_abs_path": abs_img_path,
                "modality": item.get("modality"),
                "src_content": item.get("src_content"),
            }
        )
    return resolved


def _image_to_data_uri(image_path: str) -> str:
    if not image_path or (not os.path.exists(image_path)):
        return ""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _truncate_text_html(text: str, max_chars: int) -> str:
    if not isinstance(text, str) or not text.strip():
        return '<span class="empty">(无文本)</span>'
    if max_chars <= 0 or len(text) <= max_chars:
        return html.escape(text)

    short = html.escape(text[:max_chars].rstrip())
    full = html.escape(text)
    return (
        f"{short}... "
        '<details class="more"><summary>展开</summary>'
        f'<div class="full-text">{full}</div>'
        "</details>"
    )


def _render_cell(entry: Dict[str, Any], image_width: int, max_text_chars: int) -> str:
    did = html.escape(str(entry.get("did", "")))
    found = entry.get("found", False)
    if not found:
        return f'<td class="item-cell"><div class="did">{did}</div><div class="missing">未在候选池中找到</div></td>'

    txt = entry.get("txt")
    txt_html = _truncate_text_html(txt, max_chars=max_text_chars)

    rel_img_path = entry.get("img_path")
    abs_img_path = entry.get("img_abs_path")
    img_html = "<div class=\"empty\">(无图片)</div>"
    if rel_img_path and abs_img_path and os.path.exists(abs_img_path):
        data_uri = _image_to_data_uri(abs_img_path)
        caption = html.escape(rel_img_path)
        img_html = (
            f'<img src="{data_uri}" width="{image_width}" loading="lazy" />'
            f'<div class="img-path">{caption}</div>'
        )
    elif rel_img_path:
        caption = html.escape(rel_img_path)
        img_html = f'<div class="empty">(图片路径存在但文件不存在)</div><div class="img-path">{caption}</div>'

    modality = html.escape(str(entry.get("modality") or ""))

    return (
        "<td class=\"item-cell\">"
        f"<div class=\"did\">{did}</div>"
        f"<div class=\"modality\">modality: {modality or '-'} </div>"
        f"<div class=\"txt\">{txt_html}</div>"
        f"<div class=\"img\">{img_html}</div>"
        "</td>"
    )


def _render_query_block(ds: Dict[str, Any], mbeir_root: str, image_width: int, max_text_chars: int) -> str:
    qid = html.escape(ds.get("qid", ""))
    query_txt = ds.get("query_txt") or ""
    query_img_path = ds.get("query_img_path") or ""
    query_img_path_esc = html.escape(query_img_path)

    query_txt_html = _truncate_text_html(query_txt, max_chars=max_text_chars)
    query_img_html = "<div class='empty'>(无图片)</div>"
    if query_img_path:
        abs_query_img_path = os.path.join(mbeir_root, query_img_path)
        if os.path.exists(abs_query_img_path):
            data_uri = _image_to_data_uri(abs_query_img_path)
            query_img_html = (
                f'<img src="{data_uri}" width="{image_width}" loading="lazy" />'
                f"<div class='img-path'>{query_img_path_esc}</div>"
            )
        else:
            query_img_html = f"<div class='empty'>(图片路径存在但文件不存在)</div><div class='img-path'>{query_img_path_esc}</div>"

    case_idx = ds.get("case_index_in_dataset")
    case_title = f"qid: {qid}" if case_idx is None else f"case #{case_idx} | qid: {qid}"

    return (
        "<div class='query-box'>"
        f"<div class='query-title'>{case_title}</div>"
        f"<div><b>query_txt:</b> <span class='query-txt'>{query_txt_html}</span></div>"
        f"<div><b>query_img:</b> <div class='query-img'>{query_img_html}</div></div>"
        "</div>"
    )


def render_html(report_data: Dict[str, Any], output_html: str, image_width: int, mbeir_root: str, max_text_chars: int) -> None:
    datasets = report_data.get("datasets", [])

    style = """
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { margin-bottom: 6px; }
    .meta { color: #666; margin-bottom: 20px; }
    .dataset-block { margin-bottom: 50px; }
    .query-box { border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; margin: 10px 0 16px; background: #fafafa; }
    .query-title { font-weight: bold; margin-bottom: 6px; }
    .query-txt { white-space: pre-wrap; }
    .query-img { margin-top: 6px; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    th, td { border: 1px solid #ccc; vertical-align: top; padding: 8px; }
    th.row-head { width: 130px; background: #f3f3f3; }
    td.item-cell { min-width: 220px; }
    .did { font-weight: bold; color: #2c3e50; margin-bottom: 4px; word-break: break-all; }
    .modality { color: #777; font-size: 12px; margin-bottom: 6px; }
    .txt { white-space: pre-wrap; font-size: 13px; line-height: 1.4; margin-bottom: 8px; }
    .img img { border: 1px solid #ddd; border-radius: 4px; }
    .img-path { color: #777; font-size: 12px; margin-top: 4px; word-break: break-all; }
    .empty { color: #999; font-style: italic; }
    .missing { color: #b23b3b; font-weight: bold; }
    .hint { color: #666; margin-bottom: 20px; }
    details.more { display: inline; }
    details.more summary { display: inline; color: #2563eb; cursor: pointer; margin-left: 4px; }
    .full-text { white-space: pre-wrap; margin-top: 4px; }
    """

    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>Top Cases Per Dataset - Retrieval Table</title>")
    parts.append(f"<style>{style}</style></head><body>")
    parts.append("<h1>Top Cases Per Dataset: relevant_dids / mt0_top5 / mt5_top5</h1>")
    parts.append(f"<div class='meta'>Total selected cases: {len(datasets)}</div>")
    parts.append("<div class='hint'>每个数据集多个 case；每个 case 一张表，行顺序固定：relevant_dids, mt0_top5, mt5_top5。</div>")

    for ds in datasets:
        dataset_name = html.escape(ds.get("dataset", "unknown"))

        parts.append(f"<div class='dataset-block'><h2>{dataset_name}</h2>")
        parts.append(_render_query_block(ds, mbeir_root=mbeir_root, image_width=image_width, max_text_chars=max_text_chars))

        rows = [
            ("relevant_dids", ds.get("relevant_dids_resolved", [])),
            ("mt0_top5", ds.get("mt0_top5_resolved", [])),
            ("mt5_top5", ds.get("mt5_top5_resolved", [])),
        ]

        parts.append("<table>")
        for row_name, entries in rows:
            row_name_esc = html.escape(row_name)
            parts.append(f"<tr><th class='row-head'>{row_name_esc}</th>")
            for entry in entries:
                parts.append(_render_cell(entry, image_width=image_width, max_text_chars=max_text_chars))
            parts.append("</tr>")
        parts.append("</table></div>")

    parts.append("</body></html>")

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    args = parse_args()

    payload = read_json(args.cases_json)
    cases_map = choose_cases_per_dataset(payload, cases_per_dataset=args.cases_per_dataset)
    needed_dids = collect_needed_dids(cases_map)

    cand_pool_files = list_candidate_pool_files(args.sub_mbeir_dir)
    did_lookup = build_did_lookup(needed_dids, cand_pool_files)

    datasets_out: List[Dict[str, Any]] = []
    selected_datasets_sorted = sorted(cases_map.keys())
    for dataset_name in selected_datasets_sorted:
        cases = cases_map[dataset_name]
        for idx, case in enumerate(cases, start=1):
            out = {
                "dataset": dataset_name,
                "case_index_in_dataset": idx,
                "qid": case.get("qid"),
                "query_txt": case.get("query_txt"),
                "query_img_path": case.get("query_img_path"),
                "query_modality": case.get("query_modality"),
                "relevant_dids": case.get("relevant_dids", []),
                "mt0_top5": case.get("mt0_top5", []),
                "mt5_top5": case.get("mt5_top5", []),
            }

            out["relevant_dids_resolved"] = resolve_did_items(out["relevant_dids"], did_lookup, args.mbeir_root)
            out["mt0_top5_resolved"] = resolve_did_items(out["mt0_top5"], did_lookup, args.mbeir_root)
            out["mt5_top5_resolved"] = resolve_did_items(out["mt5_top5"], did_lookup, args.mbeir_root)
            datasets_out.append(out)

    unresolved = sorted([did for did in needed_dids if did not in did_lookup])

    report = {
        "input_cases_json": args.cases_json,
        "cases_per_dataset": args.cases_per_dataset,
        "cand_pool_files": cand_pool_files,
        "selected_dataset_count": len(selected_datasets_sorted),
        "selected_case_count": len(datasets_out),
        "selected_datasets": selected_datasets_sorted,
        "needed_did_count": len(needed_dids),
        "resolved_did_count": len(needed_dids) - len(unresolved),
        "unresolved_dids": unresolved,
        "datasets": datasets_out,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    render_html(
        report,
        args.output_html,
        image_width=args.image_width,
        mbeir_root=args.mbeir_root,
        max_text_chars=args.max_text_chars,
    )

    print("Done.")
    print(f"Selected datasets: {len(selected_datasets_sorted)}")
    print(f"Selected cases: {len(datasets_out)} (target per dataset={args.cases_per_dataset})")
    print(f"Needed DIDs: {len(needed_dids)}, resolved: {len(needed_dids) - len(unresolved)}, unresolved: {len(unresolved)}")
    print(f"JSON: {args.output_json}")
    print(f"HTML: {args.output_html}")


if __name__ == "__main__":
    main()
