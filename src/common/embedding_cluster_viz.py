"""
Visualize query and candidate embeddings with dimensionality reduction and clustering.

Defaults point to the MT5 eval_finetuned embeddings under:
/data/LR1/runs/eval_finetuned/MT5/embed/JinaV4/Large/Instruct/InBatch/reason_steps_5
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless environments
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "scikit-learn is required for visualization. Install via `pip install scikit-learn`."
    ) from exc

EmbedMethod = Literal["pca", "tsne", "umap"]


def load_embeddings(path: Path, max_points: int, sample_frac: float, rng: np.random.Generator) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    emb = np.load(path)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape} for {path}")

    if 0.0 < sample_frac < 1.0:
        take = max(1, int(emb.shape[0] * sample_frac))
        indices = rng.choice(emb.shape[0], size=take, replace=False)
        emb = emb[indices]
    if max_points > 0 and emb.shape[0] > max_points:
        indices = rng.choice(emb.shape[0], size=max_points, replace=False)
        emb = emb[indices]
    return emb.astype(np.float32)


def discover_pairs(embed_root: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    Find matching (query, candidate) embedding files under the given root.

    Looks for:
    - cand pool files matching mbeir_*_cand_pool_embed.npy
    - query files matching mbeir_*_test_embed.npy

    Pairs by shared dataset key between the two filenames.
    """

    cand_map: Dict[str, Path] = {}
    query_map: Dict[str, Path] = {}

    for path in embed_root.rglob("mbeir_*_cand_pool_embed.npy"):
        name = path.stem  # e.g., mbeir_visualnews_task0_cand_pool_embed
        key = name.removeprefix("mbeir_").removesuffix("_cand_pool_embed")
        if key.lower() == "union":
            continue  # skip union pool
        cand_map[key] = path

    for path in embed_root.rglob("mbeir_*_test_embed.npy"):
        name = path.stem  # e.g., mbeir_visualnews_task0_test_embed
        key = name.removeprefix("mbeir_").removesuffix("_test_embed")
        query_map[key] = path

    pairs: Dict[str, Tuple[Path, Path]] = {}
    for key, cand_path in cand_map.items():
        query_path = query_map.get(key)
        if query_path:
            pairs[key] = (query_path, cand_path)

    return pairs


def reduce_embeddings(
    emb: np.ndarray,
    method: EmbedMethod,
    random_state: int,
    tsne_perplexity: float,
    umap_neighbors: int,
) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(emb)

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(emb)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "umap-learn is required for UMAP. Install via `pip install umap-learn`."
            ) from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        )
        return reducer.fit_transform(emb)

    raise ValueError(f"Unsupported reduction method: {method}")


def plot_scatter(
    cand_2d: np.ndarray,
    query_2d: np.ndarray,
    cluster_labels: np.ndarray | None,
    cluster_centers: np.ndarray | None,
    method: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)

    cand_kwargs = {"s": 12, "alpha": 0.55, "color": "#1f77b4", "label": "candidate"}
    query_kwargs = {"s": 18, "alpha": 0.75, "color": "#d62728", "label": "query", "marker": "^"}

    ax.scatter(cand_2d[:, 0], cand_2d[:, 1], **cand_kwargs)
    ax.scatter(query_2d[:, 0], query_2d[:, 1], **query_kwargs)

    if cluster_centers is not None:
        ax.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            marker="X",
            s=140,
            color="#111111",
            edgecolors="white",
            linewidths=1.2,
            label="kmeans centers",
            alpha=0.9,
        )
        for idx, (x, y) in enumerate(cluster_centers):
            ax.text(x, y, str(idx), fontsize=8, ha="center", va="center", color="white")

    ax.set_title(f"Embedding clustering ({method.upper()})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_combined(
    reduced: np.ndarray,
    groups: List[Tuple[str, slice, str]],
    cluster_labels: np.ndarray | None,
    cluster_centers: np.ndarray | None,
    method: str,
    output_path: Path,
) -> None:
    """
    Plot all datasets in one figure. groups holds (label, slice, marker).
    marker: '^' for query, 'o' for candidate.
    """

    fig, ax = plt.subplots(figsize=(12, 9), dpi=140)
    cmap = plt.get_cmap("tab20")

    for idx, (label, slc, marker) in enumerate(groups):
        color = cmap(idx % cmap.N)
        pts = reduced[slc]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=12 if marker == "o" else 16,
            alpha=0.65,
            color=color,
            marker=marker,
            label=label,
        )

    if cluster_centers is not None:
        ax.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            marker="X",
            s=140,
            color="#111111",
            edgecolors="white",
            linewidths=1.2,
            label="kmeans centers",
            alpha=0.9,
        )
        for cid, (x, y) in enumerate(cluster_centers):
            ax.text(x, y, str(cid), fontsize=8, ha="center", va="center", color="white")

    ax.set_title(f"Embedding clustering (combined, {method.upper()})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def describe_clusters(
    cluster_labels: np.ndarray,
    split_index: int,
    num_clusters: int,
) -> None:
    for cid in range(num_clusters):
        mask = cluster_labels == cid
        cand_count = mask[:split_index].sum()
        query_count = mask[split_index:].sum()
        total = mask.sum()
        print(f"Cluster {cid}: total={total} cand={cand_count} query={query_count}")


def process_and_plot(
    query_path: Path,
    cand_path: Path,
    output_path: Path,
    method: EmbedMethod,
    tsne_perplexity: float,
    umap_neighbors: int,
    num_clusters: int,
    seed: int,
    max_query_points: int,
    max_cand_points: int,
    sample_frac: float,
    normalize_first: bool,
    save_coords: str,
    drop_nan: bool,
) -> None:
    rng = np.random.default_rng(seed)

    query_emb = load_embeddings(Path(query_path), max_query_points, sample_frac, rng)
    cand_emb = load_embeddings(Path(cand_path), max_cand_points, sample_frac, rng)
    print(f"Loaded query embeddings: {query_emb.shape} from {query_path}")
    print(f"Loaded candidate embeddings: {cand_emb.shape} from {cand_path}")
    if normalize_first:
        query_emb = normalize(query_emb)
        cand_emb = normalize(cand_emb)

    combined = np.vstack([cand_emb, query_emb])
    if drop_nan:
        mask = ~np.isnan(combined).any(axis=1)
        dropped = combined.shape[0] - mask.sum()
        if dropped > 0:
            print(f"Dropped {dropped} rows containing NaN before reduction (single mode)")
        combined = combined[mask]
        split_index = mask[: cand_emb.shape[0]].sum()
    else:
        split_index = cand_emb.shape[0]

    reduced = reduce_embeddings(
        combined,
        method=method,
        random_state=seed,
        tsne_perplexity=tsne_perplexity,
        umap_neighbors=umap_neighbors,
    )
    cand_2d = reduced[:split_index]
    query_2d = reduced[split_index:]

    cluster_labels = None
    cluster_centers = None
    if num_clusters > 0:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
        cluster_labels = kmeans.fit_predict(reduced)
        cluster_centers = kmeans.cluster_centers_
        describe_clusters(cluster_labels, split_index, num_clusters)

    if save_coords:
        np.savez(
            save_coords,
            cand=cand_2d,
            query=query_2d,
            cluster_labels=cluster_labels,
            method=method,
        )
        print(f"Saved reduced coordinates to {save_coords}")

    plot_scatter(
        cand_2d,
        query_2d,
        cluster_labels,
        cluster_centers,
        method=method,
        output_path=Path(output_path),
    )
    print(f"Plot saved to {output_path}")


def process_batch_combined(
    pairs: Dict[str, Tuple[Path, Path]],
    method: EmbedMethod,
    tsne_perplexity: float,
    umap_neighbors: int,
    num_clusters: int,
    seed: int,
    max_query_points: int,
    max_cand_points: int,
    sample_frac: float,
    normalize_first: bool,
    save_coords: str,
    drop_nan: bool,
    output_path: Path,
) -> None:
    rng = np.random.default_rng(seed)

    cand_list: List[np.ndarray] = []
    query_list: List[np.ndarray] = []
    groups: List[Tuple[str, slice, str]] = []

    cursor = 0
    for key, (query_path, cand_path) in sorted(pairs.items()):
        q_emb = load_embeddings(query_path, max_query_points, sample_frac, rng)
        c_emb = load_embeddings(cand_path, max_cand_points, sample_frac, rng)
        if normalize_first:
            q_emb = normalize(q_emb)
            c_emb = normalize(c_emb)

        cand_list.append(c_emb)
        query_list.append(q_emb)

        start_c = cursor
        end_c = cursor + c_emb.shape[0]
        groups.append((f"{key}-cand", slice(start_c, end_c), "o"))
        cursor = end_c

        start_q = cursor
        end_q = cursor + q_emb.shape[0]
        groups.append((f"{key}-query", slice(start_q, end_q), "^"))
        cursor = end_q

        print(f"Loaded {key}: cand {c_emb.shape}, query {q_emb.shape}")

    combined = np.vstack(cand_list + query_list)
    if drop_nan:
        mask = ~np.isnan(combined).any(axis=1)
        dropped = combined.shape[0] - mask.sum()
        if dropped > 0:
            print(f"Dropped {dropped} rows containing NaN before reduction (batch combined mode)")
        combined = combined[mask]
        new_groups: List[Tuple[str, slice, str]] = []
        cursor = 0
        for label, slc, marker in groups:
            seg_mask = mask[slc.start:slc.stop]
            keep_count = int(seg_mask.sum())
            if keep_count == 0:
                continue
            new_groups.append((label, slice(cursor, cursor + keep_count), marker))
            cursor += keep_count
        groups = new_groups

    reduced = reduce_embeddings(
        combined,
        method=method,
        random_state=seed,
        tsne_perplexity=tsne_perplexity,
        umap_neighbors=umap_neighbors,
    )

    cluster_labels = None
    cluster_centers = None
    if num_clusters > 0:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
        cluster_labels = kmeans.fit_predict(reduced)
        cluster_centers = kmeans.cluster_centers_
        print(f"KMeans clusters: {num_clusters}")

    if save_coords:
        np.savez(
            save_coords,
            reduced=reduced,
            cluster_labels=cluster_labels,
            method=method,
        )
        print(f"Saved reduced coordinates to {save_coords}")

    plot_scatter_combined(
        reduced=reduced,
        groups=groups,
        cluster_labels=cluster_labels,
        cluster_centers=cluster_centers,
        method=method,
        output_path=output_path,
    )
    print(f"Combined plot saved to {output_path}")


def parse_args() -> argparse.Namespace:
    default_root = (
        "/data/LR1/runs/eval_finetuned/MT5/embed/"
        "JinaV4/Large/Instruct/InBatch/reason_steps_5"
    )
    parser = argparse.ArgumentParser(
        description="Dimensionality reduction and clustering for query/candidate embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: auto-discover all dataset pairs under embed_root and plot each",
    )
    parser.add_argument(
        "--batch_mode",
        type=str,
        choices=["separate", "combine"],
        default="combine",
        help="Batch plotting strategy: separate files per dataset or single combined figure",
    )
    parser.add_argument(
        "--embed_root",
        type=str,
        default=default_root,
        help="Root directory containing cand_pool/ and test/ embedding files",
    )
    parser.add_argument(
        "--query_path",
        type=str,
        default=os.path.join(default_root, "test", "mbeir_mscoco_task0_test_embed.npy"),
        help="Path to query embeddings (.npy)",
    )
    parser.add_argument(
        "--cand_path",
        type=str,
        default=os.path.join(default_root, "cand_pool", "mbeir_mscoco_task0_test_cand_pool_embed.npy"),
        help="Path to candidate embeddings (.npy)",
    )
    parser.add_argument(
        "--max_query_points",
        type=int,
        default=0,
        help="Max query points to sample (0 = use all)",
    )
    parser.add_argument(
        "--max_cand_points",
        type=int,
        default=0,
        help="Max candidate points to sample (0 = use all)",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Randomly sample this fraction of points from each file (0-1, applied before max_points)",
    )
    parser.add_argument(
        "--drop_nan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop any rows containing NaN before dimensionality reduction",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings before reduction",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pca", "tsne", "umap"],
        default="pca",
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE (used only when method=tsne)",
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=20,
        help="Number of neighbors for UMAP (used only when method=umap)",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=0,
        help="If >0, run KMeans with the given number of clusters",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and clustering",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(default_root, "embedding_clusters.png"),
        help="Path to save the scatter plot (single mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Batch mode: directory to save plots; defaults to embed_root",
    )
    parser.add_argument(
        "--save_coords",
        type=str,
        default="",
        help="Optional path to save reduced coordinates as .npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch:
        embed_root = Path(args.embed_root)
        pairs = discover_pairs(embed_root)
        if not pairs:
            print(f"No dataset pairs found under {embed_root}")
            return

        if args.batch_mode == "combine":
            output_dir = Path(args.output_dir) if args.output_dir else embed_root
            output_dir.mkdir(parents=True, exist_ok=True)
            outfile = output_dir / f"all_datasets_{args.method}_scatter.png"
            coords_out = output_dir / f"all_datasets_{args.method}_coords.npz" if args.save_coords else ""
            print(f"Combining {len(pairs)} dataset pairs into one plot -> {outfile}")
            process_batch_combined(
                pairs=pairs,
                method=args.method,
                tsne_perplexity=args.tsne_perplexity,
                umap_neighbors=args.umap_neighbors,
                num_clusters=args.num_clusters,
                seed=args.seed,
                max_query_points=args.max_query_points,
                max_cand_points=args.max_cand_points,
                sample_frac=args.sample_frac,
                normalize_first=args.normalize,
                drop_nan=args.drop_nan,
                save_coords=str(coords_out) if args.save_coords else "",
                output_path=outfile,
            )
        else:
            output_dir = Path(args.output_dir) if args.output_dir else embed_root
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Discovered {len(pairs)} dataset pairs. Saving plots to {output_dir}.")
            for key, (query_path, cand_path) in sorted(pairs.items()):
                outfile = output_dir / f"{key}_{args.method}_scatter.png"
                coords_out = output_dir / f"{key}_{args.method}_coords.npz" if args.save_coords else ""
                print(f"\n==> Processing {key}\n   query: {query_path}\n   cand : {cand_path}\n   out  : {outfile}")
                process_and_plot(
                    query_path=query_path,
                    cand_path=cand_path,
                    output_path=outfile,
                    method=args.method,
                    tsne_perplexity=args.tsne_perplexity,
                    umap_neighbors=args.umap_neighbors,
                    num_clusters=args.num_clusters,
                    seed=args.seed,
                    max_query_points=args.max_query_points,
                    max_cand_points=args.max_cand_points,
                    sample_frac=args.sample_frac,
                    normalize_first=args.normalize,
                    drop_nan=args.drop_nan,
                    save_coords=str(coords_out) if args.save_coords else "",
                )
    else:
        process_and_plot(
            query_path=Path(args.query_path),
            cand_path=Path(args.cand_path),
            output_path=Path(args.output),
            method=args.method,
            tsne_perplexity=args.tsne_perplexity,
            umap_neighbors=args.umap_neighbors,
            num_clusters=args.num_clusters,
            seed=args.seed,
            max_query_points=args.max_query_points,
            max_cand_points=args.max_cand_points,
            sample_frac=args.sample_frac,
            normalize_first=args.normalize,
            drop_nan=args.drop_nan,
            save_coords=args.save_coords,
        )


if __name__ == "__main__":
    main()
