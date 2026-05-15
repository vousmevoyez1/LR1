"""
Attention Analyzer: Extract, visualize and quantify thought-token attention.
=============================================================================
This module provides tools to analyze how thought tokens attend to input tokens
in the JinaEmbeddingsV4o model with multi-step reasoning.

Key Components:
1. Attention Extraction  — Forward pass with output_attentions=True
2. Heatmap Visualization — Thought tokens × Input tokens attention heatmap
3. Quantitative Metrics  — Jensen-Shannon Divergence & Attention Entropy

Usage:
    from attention_analyzer import AttentionAnalyzer
    analyzer = AttentionAnalyzer(model, processor, num_thought_tokens=5, device="cuda:0")
    result = analyzer.analyze_query(query_text="...", query_image=None)
    analyzer.plot_heatmap(result, save_path="heatmap.png")
    metrics = analyzer.compute_metrics(result)
"""

import os
import sys
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ============================================================================
# Attention Extraction
# ============================================================================

class AttentionResult:
    """
    Container for attention analysis results of a single query.
    
    Attributes:
        query_text:            Original query text string.
        query_image_path:      Path to the query image (None if text-only).
        num_thought_tokens:    Number of thought tokens K.
        input_token_ids:       Token IDs of the full input sequence (1D LongTensor).
        input_token_strings:   Decoded strings for each input token.
        thought_token_positions: Indices of thought tokens in the sequence.
        input_positions:       Indices of non-thought input tokens.
        attention_matrix:      Full head-averaged attention [SeqLen, SeqLen].
        thought_to_input_attn: Attention sub-matrix [K, InputLen] — thought tokens
                               attending to input tokens.
    """
    def __init__(self):
        self.query_text: Optional[str] = None
        self.query_image_path: Optional[str] = None
        self.num_thought_tokens: int = 0
        self.input_token_ids: Optional[torch.LongTensor] = None
        self.input_token_strings: Optional[List[str]] = None
        self.thought_token_positions: Optional[List[int]] = None
        self.input_positions: Optional[List[int]] = None
        self.attention_matrix: Optional[np.ndarray] = None
        self.thought_to_input_attn: Optional[np.ndarray] = None


class AttentionExtractor:
    """
    Extracts attention matrices from JinaEmbeddingsV4o model forward passes.
    
    This class hooks into the model to capture the last layer's attention weights
    when processing a query with thought tokens.
    """
    
    def __init__(
        self,
        model,
        processor,
        num_thought_tokens: int = 5,
        device: str = "cuda:0",
        task_label: str = "retrieval",
    ):
        """
        Args:
            model:              JinaEmbeddingsV4Model (with LoRA loaded, eval mode).
            processor:          JinaEmbeddingsV4Processor (with thought tokens added).
            num_thought_tokens: Number of thought tokens K.
            device:             Target device for inference.
            task_label:         Task identifier for LoRA adapter selection.
        """
        self.model = model
        self.processor = processor
        self.num_thought_tokens = num_thought_tokens
        self.device = device
        self.task_label = task_label
        
        # Ensure thought tokens are set up in processor
        if num_thought_tokens > 0:
            self.thought_token_ids = processor.get_thought_token_ids(num_thought_tokens)
        else:
            self.thought_token_ids = []

    @torch.no_grad()
    def extract(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
        query_image_path: Optional[str] = None,
    ) -> AttentionResult:
        """
        Run a single query through the model with output_attentions=True,
        and extract the last-layer head-averaged attention matrix.
        
        Args:
            query_text:       Query text string (or None for image-only).
            query_image:      PIL Image (or None for text-only).
            query_image_path: Path to load image from (used if query_image is None).
            
        Returns:
            AttentionResult containing the attention sub-matrix and metadata.
        """
        result = AttentionResult()
        result.query_text = query_text
        result.query_image_path = query_image_path
        result.num_thought_tokens = self.num_thought_tokens
        
        # Load image if path provided but image not
        if query_image is None and query_image_path is not None:
            full_path = query_image_path
            if not os.path.isabs(full_path):
                # Images are stored under /data/M-BEIR/ (not sub_MBEIR)
                full_path = os.path.join("/data/M-BEIR", full_path)
            query_image = Image.open(full_path).convert("RGB")
        
        # Determine modality and process inputs
        has_text = query_text is not None and query_text.strip() != ""
        has_image = query_image is not None
        
        if has_text and has_image:
            processed = self.processor.process_multimodal(
                images=[query_image],
                texts=[query_text],
                num_thought_tokens=self.num_thought_tokens,
            )
        elif has_image:
            processed = self.processor.process_images(
                images=[query_image],
                num_thought_tokens=self.num_thought_tokens,
            )
        elif has_text:
            processed = self.processor.process_texts(
                texts=[query_text],
                num_thought_tokens=self.num_thought_tokens,
            )
        else:
            raise ValueError("At least one of query_text or query_image must be provided.")
        
        # Move to device
        processed = {k: v.to(self.device) for k, v in processed.items()}
        
        input_ids = processed["input_ids"]  # [1, SeqLen]
        attention_mask = processed["attention_mask"]  # [1, SeqLen]
        
        # Save token info
        result.input_token_ids = input_ids[0].cpu()
        
        # ================================================================
        # Forward pass with output_attentions=True
        # ================================================================
        # We need to call the internal model forward, not encode_mbeir_batch,
        # to get access to the attention weights.
        
        # ================================================================
        # Navigate the model hierarchy
        # ================================================================
        # The model may be wrapped: PeftModelForFeatureExtraction -> JinaEmbeddingsV4Model -> Qwen2_5_VLModel
        # We need the JinaEmbeddingsV4Model (extends Qwen2_5_VLForConditionalGeneration)
        # and its inner Qwen2_5_VLModel for get_rope_index and forward.
        jina_model = self.model
        # Unwrap PeftModel if present
        if type(jina_model).__name__.startswith('PeftModel'):
            jina_model = jina_model.model  # -> JinaEmbeddingsV4Model
        # jina_model.model should be Qwen2_5_VLModel (has get_rope_index)
        qwen_vl_model = jina_model.model  # Qwen2_5_VLModel
        if not hasattr(qwen_vl_model, 'get_rope_index'):
            raise AttributeError(
                f"Cannot find get_rope_index on {type(qwen_vl_model).__name__}. "
                f"Expected Qwen2_5_VLModel."
            )
        
        # Prepare kwargs for forward
        forward_kwargs = {}
        if "pixel_values" in processed:
            forward_kwargs["pixel_values"] = processed["pixel_values"]
        if "image_grid_thw" in processed:
            forward_kwargs["image_grid_thw"] = processed["image_grid_thw"]
        
        # Handle pixel_values reshaping (same as get_last_hidden_states)
        # Only reshape if pixel_values is actually present and not None
        if forward_kwargs.get("pixel_values") is not None and forward_kwargs.get("image_grid_thw") is not None:
            offsets = forward_kwargs["image_grid_thw"][:, 1] * forward_kwargs["image_grid_thw"][:, 2]
            forward_kwargs["pixel_values"] = torch.cat(
                [pv[:o] for pv, o in zip(forward_kwargs["pixel_values"], offsets)], dim=0
            )
        
        # Compute position IDs using Qwen2_5_VLModel.get_rope_index
        position_ids, rope_deltas = qwen_vl_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=forward_kwargs.get("image_grid_thw", None),
            attention_mask=attention_mask,
        )
        
        # Forward with attention output enabled
        # Note: When using SDPA, setting output_attentions=True causes automatic
        # fallback to eager (manual) attention implementation.
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["output_attentions"] = True
        forward_kwargs["use_cache"] = False
        forward_kwargs["return_dict"] = True
        
        # Call Qwen2_5_VLForConditionalGeneration.forward on the JinaEmbeddingsV4Model
        # (NOT PeftModel) so that self.model(...) inside correctly dispatches to
        # Qwen2_5_VLModel.forward (which handles None pixel_values properly).
        from models.jina_v4o.jina_v4o.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        outputs = Qwen2_5_VLForConditionalGeneration.forward(
            jina_model,
            task_label=self.task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            **forward_kwargs,
        )
        
        # ================================================================
        # Extract attention from last layer, average over heads
        # ================================================================
        # outputs.attentions is a tuple of (num_layers,) tensors
        # each tensor shape: [batch, num_heads, seq_len, seq_len]
        attentions = outputs.attentions
        if attentions is None or len(attentions) == 0:
            raise RuntimeError(
                "No attention weights returned. Ensure the model supports output_attentions=True."
            )
        
        last_layer_attn = attentions[-3]  # [1, num_heads, SeqLen, SeqLen]
        # Average over heads → [1, SeqLen, SeqLen]
        #avg_attn = last_layer_attn.mean(dim=1)
        max_attn, _ = last_layer_attn.max(dim=1) 
        
        avg_attn = max_attn[0].cpu().float().numpy()  # [SeqLen, SeqLen]
        #avg_attn = avg_attn[0].cpu().float().numpy()  # [SeqLen, SeqLen]
        result.attention_matrix = avg_attn
        
        # ================================================================
        # Locate thought token and input token positions
        # ================================================================
        seq_ids = result.input_token_ids.tolist()
        thought_positions = []
        input_positions = []
        
        thought_id_set = set(self.thought_token_ids[:self.num_thought_tokens])
        
        for pos, tid in enumerate(seq_ids):
            if tid in thought_id_set:
                thought_positions.append(pos)
            else:
                # Only include positions with attention_mask=1 (non-padding)
                if attention_mask[0, pos].item() == 1:
                    input_positions.append(pos)
        
        result.thought_token_positions = thought_positions
        result.input_positions = input_positions
        
        # ================================================================
        # Slice: Thought tokens → Input tokens attention sub-matrix
        # Shape: [K, InputLen]
        # ================================================================
        if len(thought_positions) > 0 and len(input_positions) > 0:
            thought_idx = np.array(thought_positions)
            input_idx = np.array(input_positions)
            sub_matrix = avg_attn[np.ix_(thought_idx, input_idx)]
            result.thought_to_input_attn = sub_matrix
        else:
            print(f"Warning: thought_positions={len(thought_positions)}, "
                  f"input_positions={len(input_positions)}")
            result.thought_to_input_attn = None
        
        # ================================================================
        # Decode token strings for visualization
        # ================================================================
        token_strings = []
        for pos in input_positions:
            tid = seq_ids[pos]
            decoded = self.processor.tokenizer.decode([tid])
            token_strings.append(decoded)
        result.input_token_strings = token_strings
        
        return result


# ============================================================================
# Quantitative Metrics
# ============================================================================

def compute_attention_entropy(attention_row: np.ndarray) -> float:
    """
    Compute Shannon entropy of an attention distribution.
    
    H(A_k) = -Σ P(x) log P(x)
    
    Higher entropy → more uniform (dispersed) attention.
    Lower entropy → more focused (peaked) attention.
    
    Args:
        attention_row: 1D array of attention weights (should sum to ~1).
        
    Returns:
        Entropy value (nats, using natural log).
    """
    # Ensure it's a valid distribution
    p = attention_row.copy()
    p = p / (p.sum() + 1e-12)  # Re-normalize for safety
    p = p[p > 0]  # Remove zeros to avoid log(0)
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon Divergence between two distributions.
    
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    
    Measures how different two attention distributions are.
    Higher JSD → more shifted attention between adjacent thought tokens.
    
    Args:
        p, q: 1D arrays representing probability distributions.
        
    Returns:
        JSD value in [0, ln(2)] (nats).
    """
    p = p.copy() / (p.sum() + 1e-12)
    q = q.copy() / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    
    # KL divergence (with epsilon for numerical stability)
    eps = 1e-12
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def compute_all_metrics(thought_to_input_attn: np.ndarray) -> Dict:
    """
    Compute all quantitative metrics for a thought-to-input attention matrix.
    
    Args:
        thought_to_input_attn: [K, InputLen] attention sub-matrix.
        
    Returns:
        Dict with:
            - "entropy_per_thought": List[float] — entropy for each thought token
            - "jsd_between_adjacent": List[float] — JSD between consecutive thought tokens
            - "entropy_trend": str — "converging" if entropy decreases, "diverging" otherwise
            - "mean_jsd": float — average JSD across all adjacent pairs
    """
    K = thought_to_input_attn.shape[0]
    
    # --- Attention Entropy per Thought Token ---
    entropies = []
    for k in range(K):
        h = compute_attention_entropy(thought_to_input_attn[k])
        entropies.append(h)
    
    # --- JSD between Adjacent Thought Tokens ---
    jsds = []
    for k in range(K - 1):
        jsd = compute_jsd(
            thought_to_input_attn[k],
            thought_to_input_attn[k + 1],
        )
        jsds.append(jsd)
    
    # --- Determine Trend ---
    if len(entropies) >= 2:
        # Simple linear trend: compare first half vs second half mean
        first_half = np.mean(entropies[:K // 2]) if K // 2 > 0 else entropies[0]
        second_half = np.mean(entropies[K // 2:])
        entropy_trend = "converging" if second_half < first_half else "diverging"
    else:
        entropy_trend = "N/A"
    
    mean_jsd = float(np.mean(jsds)) if jsds else 0.0
    
    return {
        "entropy_per_thought": entropies,
        "jsd_between_adjacent": jsds,
        "entropy_trend": entropy_trend,
        "mean_jsd": mean_jsd,
    }

def plot_attention_heatmap(
    result: AttentionResult,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = None,
    cmap: str = "YlOrRd",
    show: bool = False,
):
    """
    Plot a heatmap of thought tokens attending to input tokens.
    (Supports extremely long sequences with unpooled image patches)
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if result.thought_to_input_attn is None:
        print("Warning: No thought-to-input attention matrix available.")
        return
    
    attn = result.thought_to_input_attn  # [K, InputLen]
    token_strings = result.input_token_strings
    K = attn.shape[0]
    SeqLen = attn.shape[1]
    
    # ====================================================================
    # 核心修改点 1：智能标签稀疏化 (Smart Sparse Labeling)
    # 不做截断，保留所有 Token。
    # 为了防止数百个 <|image_pad|> 挤在一起变成纯黑的方块，
    # 我们将图像 patch 的标签清空，仅每隔 50 个 patch 标注一次索引。
    # ====================================================================
    display_tokens = []
    img_patch_count = 0
    
    for t in token_strings:
        if "<|image_pad|>" in t or "image_pad" in t:
            # 对于局部的图像块，只隔 50 个标一个刻度，比如 p0, p50, p100...
            if img_patch_count % 50 == 0:
                display_tokens.append(f"p_{img_patch_count}")
            else:
                display_tokens.append("")  # 留空，但热力图的网格依然保留
            img_patch_count += 1
        else:
            # 处理普通的文本 Token
            t = t.replace("\n", "↵").replace(" ", "·")
            if len(t) > 12:
                t = t[:10] + ".."
            display_tokens.append(t)
            
    # ====================================================================
    # 核心修改点 2：超宽图表动态适配 (Dynamic Ultra-Wide Figsize)
    # 给予每个 Token 固定的显示宽度 (约 0.15 英寸)。
    # 如果序列有 1000 个 Token，图表宽度会自动扩展到 150 英寸。
    # ====================================================================
    y_labels = [f"Thought_{i+1}" for i in range(K)]
    
    if figsize is None:
        # 限制最大宽度为 120 英寸，防止 matplotlib 内存溢出报错
        width = min(120.0, max(12.0, SeqLen * 0.15))
        height = max(3.0, K * 0.8)
        figsize = (width, height)
    # ====================================================================
    # 🌟 新增：对比度增强逻辑 (Contrast Enhancement)
    # ====================================================================
    attn_viz = attn.copy()
    
    # 策略 1：行级 Min-Max 归一化 (Row-wise Normalization)
    # 强制让每一个 Thought (每一行) 中，最受关注的 Token 变成 1 (最深色)，
    # 最不受关注的变成 0 (最浅色)，剔除绝对值大小的干扰，只看相对分布。
    row_mins = attn_viz.min(axis=1, keepdims=True)
    row_maxs = attn_viz.max(axis=1, keepdims=True)
    attn_viz = (attn_viz - row_mins) / (row_maxs - row_mins + 1e-12)
    
    # 策略 2：非线性指数放大 (Power-law Scaling)
    # 给所有值开根号或 0.5 次方。这会将 0.01 的微弱注意力放大到 0.1，
    # 从而把“长尾”部分的微小差异在颜色上剧烈拉开。
    attn_viz = np.power(attn_viz, 0.5)  # 0.5 甚至 0.3 都可以，越小对比越夸张

    # ====================================================================
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sns.heatmap(
        attn_viz,
        xticklabels=display_tokens,
        yticklabels=y_labels,
        cmap=cmap,
        ax=ax,
        cbar=True,
        cbar_kws={"label": "Attention Weight", "shrink": 0.5}, # 颜色条缩小一点避免在长图上显得突兀
        linewidths=0.05,
        linecolor="lightgray",
    )
    
    ax.set_xlabel("Input Tokens (Text & Local Image Patches)", fontsize=12)
    ax.set_ylabel("Thought Tokens", fontsize=12)
    
    # 设置刻度文字大小
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    
    if title:
        ax.set_title(title, fontsize=13, pad=12)
    else:
        query_preview = (result.query_text or "image-only")[:60]
        ax.set_title(f"Thought Token Attention — \"{query_preview}...\"", fontsize=11, pad=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 使用较高的 DPI 保存，保证放大后图像 patch 依然清晰
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved heatmap to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
# ============================================================================
# Heatmap Visualization
# ============================================================================

# def plot_attention_heatmap(
#     result: AttentionResult,
#     save_path: Optional[str] = None,
#     title: Optional[str] = None,
#     max_input_tokens: int = 80,
#     figsize: Tuple[float, float] = None,
#     cmap: str = "YlOrRd",
#     show: bool = False,
# ):
#     """
#     Plot a heatmap of thought tokens attending to input tokens.
    
#     X-axis: Input tokens (decoded text).
#     Y-axis: Thought_1 to Thought_K.
    
#     Args:
#         result:           AttentionResult from AttentionExtractor.extract().
#         save_path:        Path to save the figure (None = don't save).
#         title:            Custom title for the plot.
#         max_input_tokens: Max number of input tokens to display (truncate if longer).
#         figsize:          Custom figure size (width, height).
#         cmap:             Colormap name for the heatmap.
#         show:             Whether to call plt.show().
#     """
#     import matplotlib
#     matplotlib.use("Agg")  # Non-interactive backend
#     import matplotlib.pyplot as plt
#     import seaborn as sns
    
#     if result.thought_to_input_attn is None:
#         print("Warning: No thought-to-input attention matrix available.")
#         return
    
#     attn = result.thought_to_input_attn  # [K, InputLen]
#     token_strings = result.input_token_strings
#     K = attn.shape[0]
    
#     # Truncate if too many input tokens (keep last max_input_tokens for visibility)
#     if attn.shape[1] > max_input_tokens:
#         # Keep the last max_input_tokens (most relevant, near thought tokens)
#         start_idx = attn.shape[1] - max_input_tokens
#         attn = attn[:, start_idx:]
#         token_strings = token_strings[start_idx:]
    
#     # Clean token strings for display (replace special chars)
#     display_tokens = []
#     for t in token_strings:
#         t = t.replace("\n", "↵").replace(" ", "·")
#         if len(t) > 12:
#             t = t[:10] + ".."
#         display_tokens.append(t)
    
#     # Y-axis labels
#     y_labels = [f"Thought_{i+1}" for i in range(K)]
    
#     # Figure size
#     if figsize is None:
#         width = max(12, len(display_tokens) * 0.25)
#         height = max(3, K * 0.8)
#         figsize = (width, height)
    
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
    
#     sns.heatmap(
#         attn,
#         xticklabels=display_tokens,
#         yticklabels=y_labels,
#         cmap=cmap,
#         ax=ax,
#         cbar=True,
#         cbar_kws={"label": "Attention Weight"},
#         linewidths=0.1,
#         linecolor="gray",
#     )
    
#     ax.set_xlabel("Input Tokens", fontsize=12)
#     ax.set_ylabel("Thought Tokens", fontsize=12)
#     ax.tick_params(axis="x", rotation=60, labelsize=7)
#     ax.tick_params(axis="y", rotation=0, labelsize=10)
    
#     if title:
#         ax.set_title(title, fontsize=13, pad=12)
#     else:
#         query_preview = (result.query_text or "image-only")[:60]
#         ax.set_title(f"Thought Token Attention — \"{query_preview}...\"", fontsize=11, pad=12)
    
#     plt.tight_layout()
    
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"Saved heatmap to: {save_path}")
    
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
    
#     return fig


def plot_metrics_summary(
    metrics: Dict,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
):
    """
    Plot entropy trend and JSD bar charts for a single query's metrics.
    
    Left panel:  Attention Entropy per Thought Token (line plot).
    Right panel: JSD between adjacent Thought Tokens (bar chart).
    
    Args:
        metrics: Dict from compute_all_metrics().
        save_path: Path to save figure.
        title: Custom suptitle.
        show: Whether to call plt.show().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    entropies = metrics["entropy_per_thought"]
    jsds = metrics["jsd_between_adjacent"]
    K = len(entropies)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # --- Left: Entropy per Thought Token ---
    x_ent = list(range(1, K + 1))
    ax1.plot(x_ent, entropies, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.set_xlabel("Thought Token Index", fontsize=11)
    ax1.set_ylabel("Attention Entropy (nats)", fontsize=11)
    ax1.set_title(f"Attention Entropy (trend: {metrics['entropy_trend']})", fontsize=12)
    ax1.set_xticks(x_ent)
    ax1.set_xticklabels([f"T{i}" for i in x_ent])
    ax1.grid(True, alpha=0.3)
    
    # --- Right: JSD between Adjacent Thought Tokens ---
    if jsds:
        x_jsd = list(range(len(jsds)))
        labels_jsd = [f"T{i+1}→T{i+2}" for i in range(len(jsds))]
        ax2.bar(x_jsd, jsds, color="coral", edgecolor="darkred", alpha=0.8)
        ax2.set_xlabel("Adjacent Thought Token Pair", fontsize=11)
        ax2.set_ylabel("Jensen-Shannon Divergence", fontsize=11)
        ax2.set_title(f"Attention Shift (mean JSD: {metrics['mean_jsd']:.4f})", fontsize=12)
        ax2.set_xticks(x_jsd)
        ax2.set_xticklabels(labels_jsd, fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "Need K ≥ 2 for JSD", ha="center", va="center", fontsize=12)
        ax2.set_title("Attention Shift (JSD)", fontsize=12)
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved metrics plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def print_metrics_table(metrics: Dict, qid: str = ""):
    """
    Print metrics in a readable table format.
    
    Args:
        metrics: Dict from compute_all_metrics().
        qid: Query ID for display.
    """
    K = len(metrics["entropy_per_thought"])
    
    print(f"\n{'=' * 60}")
    if qid:
        print(f"  Metrics for Query: {qid}")
    print(f"{'=' * 60}")
    
    print(f"\n  Attention Entropy per Thought Token:")
    print(f"  {'Token':<12} {'Entropy (nats)':<18}")
    print(f"  {'-'*30}")
    for i, h in enumerate(metrics["entropy_per_thought"]):
        print(f"  Thought_{i+1:<4} {h:.6f}")
    print(f"  Trend: {metrics['entropy_trend']}")
    
    print(f"\n  JSD between Adjacent Thought Tokens:")
    print(f"  {'Pair':<15} {'JSD':<18}")
    print(f"  {'-'*33}")
    for i, jsd in enumerate(metrics["jsd_between_adjacent"]):
        print(f"  T{i+1} → T{i+2}     {jsd:.6f}")
    print(f"  Mean JSD: {metrics['mean_jsd']:.6f}")
    print(f"{'=' * 60}\n")
