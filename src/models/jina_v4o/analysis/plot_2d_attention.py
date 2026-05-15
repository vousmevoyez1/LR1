# """
# 2D Thought-Token Image Attention Dynamics
# =========================================

# From selected cases JSON, pick one image-containing query per target task and
# plot how image attention changes across thought tokens on the original 2D image.

# Target tasks (default):
# 	- visualnews_task3
# 	- mscoco_task3
# 	- fashion200k_task3
# 	- oven_task6
# 	- infoseek_task6

# Output per task:
# 	- Row 1: Original + Thought-1..K attention overlays
# 	- Row 2: Delta overlays (Thought i - Thought i-1), enhanced for visibility
# """

# import os
# import sys
# import json
# import argparse
# from typing import Dict, List, Tuple, Optional

# import numpy as np
# import torch
# from PIL import Image

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt


# # Make local analysis imports work when script is launched from anywhere.
# ANALYSIS_DIR = "/data/LR1/src/models/jina_v4o/analysis"
# SRC_DIR = "/data/LR1/src"
# if ANALYSIS_DIR not in sys.path:
# 	sys.path.insert(0, ANALYSIS_DIR)
# if SRC_DIR not in sys.path:
# 	sys.path.insert(0, SRC_DIR)

# from attention_analyzer import AttentionExtractor
# from run_analysis import load_model_for_analysis, build_query_text_with_instruction


# DEFAULT_CASES_JSON = "/data/LR1/src/models/jina_v4o/analysis/output/selected_cases_all_single_mt5_recall_mt0_miss.json"
# DEFAULT_OUTPUT_DIR = "/data/LR1/src/models/jina_v4o/analysis/output/2d_attention_dynamics"
# DEFAULT_CHECKPOINT = "/data/LR1/checkpointMTO5/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth"
# DEFAULT_TASKS = [
# 	"visualnews_task3",
# 	"mscoco_task3",
# 	"fashion200k_task3",
# 	"oven_task6",
# 	"infoseek_task6",
# ]


# def _resolve_image_path(rel_or_abs_path: str) -> str:
# 	if os.path.isabs(rel_or_abs_path):
# 		return rel_or_abs_path
# 	# M-BEIR image root
# 	return os.path.join("/data/M-BEIR", rel_or_abs_path)


# def _choose_one_case_per_task(cases: List[Dict], tasks: List[str]) -> Dict[str, Dict]:
# 	"""Choose one image-containing case per task, prioritizing larger delta_rank."""
# 	selected = {}
# 	for task in tasks:
# 		pool = [
# 			c for c in cases
# 			if c.get("dataset") == task
# 			and c.get("query_img_path")
# 			and "image" in str(c.get("query_modality", ""))
# 		]
# 		if not pool:
# 			continue

# 		def sort_key(x):
# 			# Prefer stronger improvement and available image.
# 			return (int(x.get("delta_rank", -1)), int(x.get("mt5_effective_rank", 10**9) * -1))

# 		pool.sort(key=sort_key, reverse=True)
# 		selected[task] = pool[0]
# 	return selected


# def _get_image_token_id(model) -> int:
# 	if hasattr(model, "config") and hasattr(model.config, "image_token_id"):
# 		return int(model.config.image_token_id)
# 	# fallback for wrapped model
# 	if hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "image_token_id"):
# 		return int(model.model.config.image_token_id)
# 	raise RuntimeError("Cannot find image_token_id from model config")


# def _extract_image_attention_maps(
# 	result,
# 	image_token_id: int,
# 	image_grid_thw: torch.Tensor,
# ) -> np.ndarray:
# 	"""
# 	Build per-thought 2D image attention maps.

# 	Returns:
# 		maps: [K, H, W]
# 	"""
# 	input_ids = result.input_token_ids.cpu().numpy().tolist()
# 	thought_positions = result.thought_token_positions or []
# 	if len(thought_positions) == 0:
# 		raise RuntimeError("No thought token positions found")

# 	image_positions = [i for i, tid in enumerate(input_ids) if int(tid) == int(image_token_id)]
# 	if len(image_positions) == 0:
# 		raise RuntimeError("No image token positions found in sequence")

# 	attn = result.attention_matrix  # [seq, seq], head-averaged
# 	t, h, w = [int(x) for x in image_grid_thw[0].tolist()]
# 	expected = t * h * w

# 	usable = min(len(image_positions), expected)
# 	if usable <= 0:
# 		raise RuntimeError("Invalid image token count / grid shape")

# 	image_positions = image_positions[:usable]
# 	if expected != usable:
# 		# Fallback: infer a square-ish map from actual image token count.
# 		side = int(np.sqrt(usable))
# 		side = max(1, side)
# 		usable = side * side
# 		image_positions = image_positions[:usable]
# 		t, h, w = 1, side, side

# 	maps = []
# 	for p in thought_positions:
# 		vec = attn[p, image_positions].astype(np.float64)
# 		# Normalize this thought's image attention distribution.
# 		vec = vec / (vec.sum() + 1e-12)
# 		grid = vec.reshape(t, h, w)
# 		map_2d = grid.mean(axis=0)
# 		maps.append(map_2d)

# 	return np.stack(maps, axis=0)  # [K, H, W]


# def _enhance_positive_map(m: np.ndarray, gamma: float = 0.4) -> np.ndarray:
# 	"""Robust normalization + gamma enhancement for stronger visual contrast."""
# 	p_low = np.percentile(m, 5)
# 	p_high = np.percentile(m, 99)
# 	denom = max(p_high - p_low, 1e-12)
# 	x = np.clip((m - p_low) / denom, 0.0, 1.0)
# 	x = np.power(x, gamma)
# 	return x


# def _enhance_delta_map(d: np.ndarray, boost: float = 2.4) -> np.ndarray:
# 	"""Signed enhancement for change maps to make differences more obvious."""
# 	# z-score like scaling with robust denominator
# 	scale = np.percentile(np.abs(d), 95)
# 	scale = max(scale, 1e-12)
# 	x = np.clip(d / scale, -1.0, 1.0)
# 	x = np.sign(x) * np.power(np.abs(x), 0.6)
# 	x = np.clip(x * boost, -1.0, 1.0)
# 	return x


# def _plot_case_dynamics(
# 	case: Dict,
# 	image_rgb: np.ndarray,
# 	thought_maps: np.ndarray,
# 	save_path: str,
# ):
# 	"""Plot original image + thought overlays + delta overlays."""
# 	K = thought_maps.shape[0]
# 	cols = K + 1

# 	fig, axes = plt.subplots(2, cols, figsize=(3.4 * cols, 7.2))

# 	for ax in axes.flatten():
# 		ax.axis("off")

# 	# Row 1, col 0: original image
# 	axes[0, 0].imshow(image_rgb)
# 	axes[0, 0].set_title("Original")

# 	# Row 2, col 0: metadata text block
# 	qid = case.get("qid", "?")
# 	ds = case.get("dataset", "?")
# 	qtxt = (case.get("query_txt") or "").strip()
# 	if len(qtxt) > 110:
# 		qtxt = qtxt[:107] + "..."
# 	axes[1, 0].text(
# 		0.02,
# 		0.95,
# 		f"dataset: {ds}\nqid: {qid}\nquery: {qtxt}",
# 		va="top",
# 		ha="left",
# 		fontsize=10,
# 		wrap=True,
# 	)

# 	# Row 1: per-thought overlays
# 	for i in range(K):
# 		m = _enhance_positive_map(thought_maps[i])
# 		ax = axes[0, i + 1]
# 		ax.imshow(image_rgb)
# 		im = ax.imshow(m, cmap="turbo", alpha=0.72, interpolation="bilinear")
# 		ax.set_title(f"Thought {i+1}")

# 	# Row 2: deltas between adjacent thoughts
# 	for i in range(1, K):
# 		d = thought_maps[i] - thought_maps[i - 1]
# 		d = _enhance_delta_map(d)
# 		ax = axes[1, i]
# 		ax.imshow(image_rgb)
# 		im_delta = ax.imshow(d, cmap="seismic", alpha=0.70, vmin=-1.0, vmax=1.0, interpolation="bilinear")
# 		ax.set_title(f"Δ T{i+1} - T{i}")

# 	# Keep last panel for legend/color cue if available
# 	if K >= 2:
# 		ax_last = axes[1, K]
# 		grad = np.linspace(-1, 1, 256).reshape(1, -1)
# 		ax_last.imshow(grad, cmap="seismic", aspect="auto", vmin=-1, vmax=1)
# 		ax_last.set_title("Δ scale")
# 		ax_last.set_yticks([])
# 		ax_last.set_xticks([0, 128, 255])
# 		ax_last.set_xticklabels(["-", "0", "+"])
# 		ax_last.axis("on")

# 	fig.suptitle(f"2D Image Attention Dynamics | {ds} | {qid}", fontsize=14)
# 	fig.tight_layout(rect=[0, 0.02, 1, 0.96])

# 	os.makedirs(os.path.dirname(save_path), exist_ok=True)
# 	fig.savefig(save_path, dpi=220)
# 	plt.close(fig)


# def main():
# 	parser = argparse.ArgumentParser(description="Plot 2D thought-token image attention dynamics per task")
# 	parser.add_argument("--cases_json", type=str, default=DEFAULT_CASES_JSON)
# 	parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
# 	parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
# 	parser.add_argument("--num_thought_tokens", type=int, default=5)
# 	parser.add_argument("--device", type=str, default="cuda:0")
# 	parser.add_argument(
# 		"--tasks",
# 		type=str,
# 		default=",".join(DEFAULT_TASKS),
# 		help="Comma-separated task names",
# 	)
# 	args = parser.parse_args()

# 	tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
# 	os.makedirs(args.output_dir, exist_ok=True)

# 	with open(args.cases_json, "r", encoding="utf-8") as f:
# 		payload = json.load(f)
# 	all_cases = payload.get("cases", payload)

# 	selected = _choose_one_case_per_task(all_cases, tasks)
# 	if not selected:
# 		raise RuntimeError("No valid image-containing cases found for requested tasks")

# 	print("Selected cases:")
# 	for t in tasks:
# 		c = selected.get(t)
# 		if c is None:
# 			print(f"  - {t}: NOT FOUND")
# 		else:
# 			print(f"  - {t}: {c.get('qid')} ({c.get('query_img_path')})")

# 	model, processor = load_model_for_analysis(
# 		checkpoint_path=args.checkpoint,
# 		num_thought_tokens=args.num_thought_tokens,
# 		device=args.device,
# 	)
# 	extractor = AttentionExtractor(
# 		model=model,
# 		processor=processor,
# 		num_thought_tokens=args.num_thought_tokens,
# 		device=args.device,
# 		task_label="retrieval",
# 	)

# 	image_token_id = _get_image_token_id(model)
# 	summary = {
# 		"tasks": tasks,
# 		"selected": {},
# 		"outputs": [],
# 		"num_thought_tokens": args.num_thought_tokens,
# 	}

# 	for task in tasks:
# 		case = selected.get(task)
# 		if case is None:
# 			continue

# 		qid = case.get("qid", "unknown")
# 		img_path = _resolve_image_path(case["query_img_path"])
# 		image = Image.open(img_path).convert("RGB")
# 		image_rgb = np.asarray(image)

# 		query_text = build_query_text_with_instruction(case)

# 		result = extractor.extract(
# 			query_text=query_text if query_text.strip() else None,
# 			query_image=image,
# 			query_image_path=img_path,
# 		)

# 		# Rebuild processed batch to get image grid dimensions.
# 		proc = processor.process_multimodal(
# 			images=[image],
# 			texts=[query_text],
# 			num_thought_tokens=args.num_thought_tokens,
# 		)
# 		if "image_grid_thw" not in proc:
# 			raise RuntimeError(f"image_grid_thw missing for task={task}, qid={qid}")

# 		thought_maps = _extract_image_attention_maps(
# 			result=result,
# 			image_token_id=image_token_id,
# 			image_grid_thw=proc["image_grid_thw"],
# 		)

# 		out_name = f"{task}__{qid.replace(':', '_')}__2d_dynamics.png"
# 		out_path = os.path.join(args.output_dir, out_name)
# 		_plot_case_dynamics(
# 			case=case,
# 			image_rgb=image_rgb,
# 			thought_maps=thought_maps,
# 			save_path=out_path,
# 		)

# 		summary["selected"][task] = {
# 			"qid": qid,
# 			"query_img_path": case.get("query_img_path"),
# 			"query_txt": case.get("query_txt", ""),
# 		}
# 		summary["outputs"].append(out_path)
# 		print(f"Saved: {out_path}")

# 	summary_path = os.path.join(args.output_dir, "selection_and_outputs.json")
# 	with open(summary_path, "w", encoding="utf-8") as f:
# 		json.dump(summary, f, indent=2, ensure_ascii=False)
# 	print(f"Summary saved: {summary_path}")


# if __name__ == "__main__":
# 	main()
"""
2D Thought-Token Image Attention Dynamics
=========================================

From selected cases JSON, pick the top-K image-containing queries per target task and
plot how image attention changes across thought tokens on the original 2D image.

Target tasks (default):
    - visualnews_task3
    - mscoco_task3
    - fashion200k_task3
    - oven_task6
    - infoseek_task6

Output per task:
    - Row 1: Original + Thought-1..K attention overlays
    - Row 2: Delta overlays (Thought i - Thought i-1), enhanced for visibility
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make local analysis imports work when script is launched from anywhere.
ANALYSIS_DIR = "/data/LR1/src/models/jina_v4o/analysis"
SRC_DIR = "/data/LR1/src"
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from attention_analyzer import AttentionExtractor
from run_analysis import load_model_for_analysis, build_query_text_with_instruction

DEFAULT_CASES_JSON = "/data/LR1/src/models/jina_v4o/analysis/output/selected_cases_all_union_mt5_recall_mt0_miss.json"
DEFAULT_OUTPUT_DIR = "/data/LR1/src/models/jina_v4o/analysis/output/2d_attention_dynamics2"
DEFAULT_CHECKPOINT = "/data/LR1/checkpointMTO5/jina_v4o/Large/Instruct/InBatch/jina_v4o_epoch_2.pth"
DEFAULT_TASKS = [
    "visualnews_task3",
    "mscoco_task3",
    "fashion200k_task3",
    "oven_task6",
    "infoseek_task6",
]


def _resolve_image_path(rel_or_abs_path: str) -> str:
    if os.path.isabs(rel_or_abs_path):
        return rel_or_abs_path
    # M-BEIR image root
    return os.path.join("/data/M-BEIR", rel_or_abs_path)


# 🌟 修改点 1：返回 Top-K 个 cases 的列表，而不是只返回 1 个
def _choose_top_k_cases_per_task(cases: List[Dict], tasks: List[str], top_k: int = 3) -> Dict[str, List[Dict]]:
    """Choose top K image-containing cases per task, prioritizing larger delta_rank."""
    selected = {}
    for task in tasks:
        pool = [
            c for c in cases
            if c.get("dataset") == task
            and c.get("query_img_path")
            and "image" in str(c.get("query_modality", ""))
        ]
        if not pool:
            continue

        def sort_key(x):
            # Prefer stronger improvement and available image.
            return (int(x.get("delta_rank", -1)), int(x.get("mt5_effective_rank", 10**9) * -1))

        pool.sort(key=sort_key, reverse=True)
        selected[task] = pool[:top_k]  # 截取前 K 个
    return selected


def _get_image_token_id(model) -> int:
    if hasattr(model, "config") and hasattr(model.config, "image_token_id"):
        return int(model.config.image_token_id)
    # fallback for wrapped model
    if hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "image_token_id"):
        return int(model.model.config.image_token_id)
    raise RuntimeError("Cannot find image_token_id from model config")


def _extract_image_attention_maps(
    result,
    image_token_id: int,
    image_grid_thw: torch.Tensor,
) -> np.ndarray:
    """Build per-thought 2D image attention maps."""
    input_ids = result.input_token_ids.cpu().numpy().tolist()
    thought_positions = result.thought_token_positions or []
    if len(thought_positions) == 0:
        raise RuntimeError("No thought token positions found")

    image_positions = [i for i, tid in enumerate(input_ids) if int(tid) == int(image_token_id)]
    if len(image_positions) == 0:
        raise RuntimeError("No image token positions found in sequence")

    attn = result.attention_matrix  # [seq, seq], head-averaged
    t, h, w = [int(x) for x in image_grid_thw[0].tolist()]
    expected = t * h * w

    usable = min(len(image_positions), expected)
    if usable <= 0:
        raise RuntimeError("Invalid image token count / grid shape")

    image_positions = image_positions[:usable]
    if expected != usable:
        # Fallback: infer a square-ish map from actual image token count.
        side = int(np.sqrt(usable))
        side = max(1, side)
        usable = side * side
        image_positions = image_positions[:usable]
        t, h, w = 1, side, side

    maps = []
    for p in thought_positions:
        vec = attn[p, image_positions].astype(np.float64)
        # Normalize this thought's image attention distribution.
        # ==========================================
        # 🌟 核心修改 1：指数锐化 (Sharpening)
        # 把原始权重取 2 次方或 3 次方，拉开绝对差距
        # ==========================================
        vec = np.power(vec, 2.5)  # 尝试 2.0 到 3.0 之间的值
        vec = vec / (vec.sum() + 1e-12)
        grid = vec.reshape(t, h, w)
        map_2d = grid.mean(axis=0)
        maps.append(map_2d)

    return np.stack(maps, axis=0)  # [K, H, W]


def _enhance_positive_map(m: np.ndarray, gamma: float = 0.4) -> np.ndarray:
    """Robust normalization + gamma enhancement for stronger visual contrast."""
    p_low = np.percentile(m, 5)
    p_high = np.percentile(m, 99)
    denom = max(p_high - p_low, 1e-12)
    x = np.clip((m - p_low) / denom, 0.0, 1.0)
    x = np.power(x, gamma)
    return x


def _enhance_delta_map(d: np.ndarray, boost: float = 2.4) -> np.ndarray:
    """Signed enhancement for change maps to make differences more obvious."""
    # z-score like scaling with robust denominator
    scale = np.percentile(np.abs(d), 95)
    scale = max(scale, 1e-12)
    x = np.clip(d / scale, -1.0, 1.0)
    x = np.sign(x) * np.power(np.abs(x), 0.6)
    x = np.clip(x * boost, -1.0, 1.0)
    return x

def _apply_transparent_cmap(
    m: np.ndarray, 
    target_shape: Tuple[int, int],
    cmap_name: str = "turbo", 
    base_alpha: float = 0.8,
    is_delta: bool = False
) -> np.ndarray:
    """
    将热力图缩放至原图尺寸，应用色谱，并实现动态透明度（强度越高越不透明）。
    """
    import matplotlib.cm
    
    h, w = target_shape
    
    # 1. 双线性插值，将小尺寸热力图无损拉伸到原图分辨率
    m_pil = Image.fromarray(m.astype(np.float32))
    m_resized = np.array(m_pil.resize((w, h), resample=Image.BILINEAR))
    
    # 获取色谱 (兼容新老版本的 matplotlib)
    try:
        cmap_matplotlib = matplotlib.colormaps[cmap_name]
    except AttributeError:
        cmap_matplotlib = matplotlib.cm.get_cmap(cmap_name)
    
    if is_delta:
        # 对于 Delta 变化图 (范围 -1.0 到 1.0)
        # 将 -1~1 归一化到 0~1 以便映射颜色
        m_norm = (np.clip(m_resized, -1.0, 1.0) + 1.0) / 2.0
        rgba_image = cmap_matplotlib(m_norm)
        # Alpha 透明度取决于变化的绝对值大小：变化越大越不透明
        alpha_mask = np.abs(m_resized) * base_alpha
    else:
        # 对于正常的 Positive 热力图 (范围 0.0 到 1.0)
        rgba_image = cmap_matplotlib(m_resized)
        # Alpha 透明度直接取决于注意力强度
        alpha_mask = m_resized * base_alpha
        
    # 2. 🌟 核心魔法：注入动态透明度
    rgba_image[:, :, 3] = np.clip(alpha_mask, 0.0, 1.0)
    
    return rgba_image


def _plot_case_dynamics(
    case: Dict,
    image_rgb: np.ndarray,
    thought_maps: np.ndarray,
    save_path: str,
):
    """Plot original image + thought overlays + delta overlays."""
    K = thought_maps.shape[0]
    cols = K + 1

    fig, axes = plt.subplots(2, cols, figsize=(3.4 * cols, 7.2))

    for ax in axes.flatten():
        ax.axis("off")

    # Row 1, col 0: original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original")

    # Row 2, col 0: metadata text block
    qid = case.get("qid", "?")
    ds = case.get("dataset", "?")
    qtxt = (case.get("query_txt") or "").strip()
    if len(qtxt) > 110:
        qtxt = qtxt[:107] + "..."
    axes[1, 0].text(
        0.02,
        0.95,
        f"dataset: {ds}\nqid: {qid}\nquery: {qtxt}",
        va="top",
        ha="left",
        fontsize=10,
        wrap=True,
    )
    
    # 提取原图的宽高用于拉伸对齐
    img_shape = image_rgb.shape[:2]

    # Row 1: per-thought overlays
    baseline_map = thought_maps.mean(axis=0)
    for i in range(K):
        # 减去公共基线，突出当前 Thought 独有的关注点
        unique_map = thought_maps[i] - baseline_map 
        unique_map = np.clip(unique_map, 0, None)
        
        # 归一化与增强
        m = _enhance_positive_map(unique_map, gamma=0.3) 
        
        # 生成带动态透明度的 RGBA 覆盖层
        rgba_overlay = _apply_transparent_cmap(
            m, target_shape=img_shape, cmap_name="turbo", base_alpha=0.85
        )

        ax = axes[0, i + 1]
        ax.imshow(image_rgb) # 先画高清原图
        ax.imshow(rgba_overlay) # 再严丝合缝地盖上带有渐变透明度的热力图
        ax.set_title(f"Thought {i+1} (Unique)")

    # Row 2: deltas between adjacent thoughts
    for i in range(1, K):
        d = thought_maps[i] - thought_maps[i - 1]
        d = _enhance_delta_map(d)
        
        # 生成发散色谱的 RGBA 覆盖层 (带有双向动态透明度)
        rgba_delta = _apply_transparent_cmap(
            d, target_shape=img_shape, cmap_name="seismic", base_alpha=0.85, is_delta=True
        )
        
        ax = axes[1, i]
        ax.imshow(image_rgb)
        ax.imshow(rgba_delta)
        ax.set_title(f"Δ T{i+1} - T{i}")

    # Keep last panel for legend/color cue if available
    if K >= 2:
        ax_last = axes[1, K]
        grad = np.linspace(-1, 1, 256).reshape(1, -1)
        ax_last.imshow(grad, cmap="seismic", aspect="auto", vmin=-1, vmax=1)
        ax_last.set_title("Δ scale")
        ax_last.set_yticks([])
        ax_last.set_xticks([0, 128, 255])
        ax_last.set_xticklabels(["-", "0", "+"])
        ax_last.axis("on")

    fig.suptitle(f"2D Image Attention Dynamics | {ds} | {qid}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot 2D thought-token image attention dynamics per task")
    parser.add_argument("--cases_json", type=str, default=DEFAULT_CASES_JSON)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num_thought_tokens", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    # 🌟 新增参数：每个任务画几张图
    parser.add_argument("--top_k_per_task", type=int, default=6, help="Number of top cases to plot per task")
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cases_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    all_cases = payload.get("cases", payload)

    # 获取 Top-K 案例
    selected = _choose_top_k_cases_per_task(all_cases, tasks, args.top_k_per_task)
    if not selected:
        raise RuntimeError("No valid image-containing cases found for requested tasks")

    print(f"Selected up to {args.top_k_per_task} cases per task:")
    for t in tasks:
        c_list = selected.get(t, [])
        if not c_list:
            print(f"  - {t}: NOT FOUND")
        else:
            for idx, c in enumerate(c_list):
                print(f"  - {t} [Top {idx+1}]: {c.get('qid')} ({c.get('query_img_path')})")

    model, processor = load_model_for_analysis(
        checkpoint_path=args.checkpoint,
        num_thought_tokens=args.num_thought_tokens,
        device=args.device,
    )
    extractor = AttentionExtractor(
        model=model,
        processor=processor,
        num_thought_tokens=args.num_thought_tokens,
        device=args.device,
        task_label="retrieval",
    )

    image_token_id = _get_image_token_id(model)
    summary = {
        "tasks": tasks,
        "selected": {t: [] for t in tasks}, # 初始化为空列表
        "outputs": [],
        "num_thought_tokens": args.num_thought_tokens,
    }

    # 🌟 修改点 2：双层循环，遍历每个任务下的所有选中的 Case
    for task in tasks:
        cases_for_task = selected.get(task, [])
        
        for idx, case in enumerate(cases_for_task):
            qid = case.get("qid", "unknown")
            img_path = _resolve_image_path(case["query_img_path"])
            image = Image.open(img_path).convert("RGB")
            image_rgb = np.asarray(image)

            query_text = build_query_text_with_instruction(case)

            print(f"Processing: {task} - Top {idx+1} ({qid})")

            result = extractor.extract(
                query_text=query_text if query_text.strip() else None,
                query_image=image,
                query_image_path=img_path,
            )

            # Rebuild processed batch to get image grid dimensions.
            proc = processor.process_multimodal(
                images=[image],
                texts=[query_text],
                num_thought_tokens=args.num_thought_tokens,
            )
            if "image_grid_thw" not in proc:
                raise RuntimeError(f"image_grid_thw missing for task={task}, qid={qid}")

            thought_maps = _extract_image_attention_maps(
                result=result,
                image_token_id=image_token_id,
                image_grid_thw=proc["image_grid_thw"],
            )

            # 🌟 修改点 3：文件名加上 topX 标识，避免被覆盖
            out_name = f"{task}_top{idx+1}_{qid.replace(':', '_')}__2d_dynamics.png"
            out_path = os.path.join(args.output_dir, out_name)
            
            _plot_case_dynamics(
                case=case,
                image_rgb=image_rgb,
                thought_maps=thought_maps,
                save_path=out_path,
            )

            summary["selected"][task].append({
                "rank": idx + 1,
                "qid": qid,
                "query_img_path": case.get("query_img_path"),
                "query_txt": case.get("query_txt", ""),
            })
            summary["outputs"].append(out_path)
            print(f"  -> Saved: {out_path}")

    summary_path = os.path.join(args.output_dir, "selection_and_outputs.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()