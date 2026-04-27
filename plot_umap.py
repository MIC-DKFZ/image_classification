#!/usr/bin/env python3
"""
Plot a UMAP from a precomputed HDF5 feature file produced by extract_features.py.

Usage:
    python plot_umap.py \
        --h5_path precomputed_features/agg_joint_vit_base_patch16_224_aid_val_size224_float16.h5

    python plot_umap.py \
        --h5_path precomputed_features/features.h5 \
        --title "ViT-B / AID (val)" \
        --output_path umap_plots/aid_val.png
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from glovita.configs.cli import parse_cli

try:
    import umap as umap_module
except ImportError as exc:
    raise ImportError("Install umap-learn: pip install umap-learn") from exc


MAX_SAMPLES = 5000
MAX_PER_CLASS = 200


class PlotUmapConfig(BaseModel):
    h5_path: Path
    title: str | None = None
    output_path: Path | None = None
    max_samples: int = MAX_SAMPLES
    max_per_class: int = MAX_PER_CLASS
    seed: int = 42
    dpi: int = 200
    n_neighbors: int = 15
    min_dist: float = 0.1


def load_and_subsample(h5_path: Path, max_samples: int, max_per_class: int, seed: int):
    with h5py.File(h5_path, "r") as f:
        if "features" not in f or "labels" not in f:
            raise KeyError(f"{h5_path} must contain 'features' and 'labels' datasets.")
        features = f["features"][:].astype(np.float32)
        labels = f["labels"][:]

    if features.ndim != 2:
        raise ValueError(
            f"{h5_path} contains features with shape {features.shape}. "
            "plot_umap.py expects instance-level features with shape (N, D)."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"{h5_path} contains labels with shape {labels.shape}. "
            "plot_umap.py expects labels with shape (N,)."
        )
    if len(features) != len(labels):
        raise ValueError(
            f"{h5_path} contains {len(features)} features but {len(labels)} labels."
        )

    if len(labels) <= max_samples:
        return features, labels

    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        keep.append(rng.choice(idx, min(len(idx), max_per_class), replace=False))
    keep = np.concatenate(keep)
    if len(keep) > max_samples:
        keep = rng.choice(keep, max_samples, replace=False)
    return features[keep], labels[keep]


def fit_umap(features: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    reducer = umap_module.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
    )
    return reducer.fit_transform(features)


def colormap_for(n: int):
    name = "tab10" if n <= 10 else "tab20" if n <= 20 else "turbo"
    return plt.get_cmap(name, n)


def plot_umap(xy: np.ndarray, labels: np.ndarray, title: str, output_path: Path, dpi: int) -> None:
    n_cls = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=labels,
        cmap=colormap_for(n_cls),
        vmin=0,
        vmax=max(n_cls - 1, 1),
        s=6,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def _resolve_output_path(config: PlotUmapConfig) -> Path:
    if config.output_path is not None:
        return config.output_path
    return Path("umap_plots/single") / f"{config.h5_path.stem}.png"


def _resolve_title(config: PlotUmapConfig, num_samples: int, num_classes: int) -> str:
    prefix = config.title or config.h5_path.stem
    return f"{prefix}\nn={num_samples:,}  classes={num_classes}"


def main(config: PlotUmapConfig) -> None:
    if not config.h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {config.h5_path}")

    print(f"H5 file: {config.h5_path}")
    print("Loading embeddings...")
    features, labels = load_and_subsample(
        config.h5_path, config.max_samples, config.max_per_class, config.seed
    )
    n_cls = len(np.unique(labels))
    print(f"  {len(labels):,} samples, {n_cls} classes")

    print("Fitting UMAP...")
    xy = fit_umap(features, config.n_neighbors, config.min_dist, config.seed)

    title = _resolve_title(config, len(labels), n_cls)
    out = _resolve_output_path(config)
    plot_umap(xy, labels, title, out, config.dpi)


if __name__ == "__main__":
    main(parse_cli(PlotUmapConfig))
