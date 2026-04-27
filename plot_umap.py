#!/usr/bin/env python3
"""
Plot a UMAP for a single (model, dataset) combination.

Resolves the HDF5 filename from the model/data config objects (matching the
naming convention used by extract_features.py), loads embeddings, fits UMAP,
and saves a scatter plot PNG.

If no matching HDF5 is found, run extract_features.py first to produce it.

Usage:
    python plot_umap.py \
        --model.encoder.encoder_type timm \
        --model.encoder.type vit_base_patch16_224 \
        --data.dataset aid \
        --data.data_root_dir /data/aid \
        --embeddings_dir precomputed_features

    python plot_umap.py \
        --model.encoder.encoder_type dinov2 \
        --model.encoder.type dinov2_vitb14 \
        --data.dataset flowers102 \
        --data.data_root_dir /data/flowers102 \
        --split test \
        --embeddings_dir precomputed_features
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Literal, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from glovita.configs.cli import parse_cli
from glovita.configs.data import DataConfig
from glovita.configs.model import ModelConfig
from extract_features import _serialize_model_name

try:
    import umap as umap_module
except ImportError:
    raise ImportError("Install umap-learn: pip install umap-learn")


MAX_SAMPLES = 5000
MAX_PER_CLASS = 200


class PlotUmapConfig(BaseModel):
    model: Optional[ModelConfig] = None
    data: Optional[DataConfig] = None
    embeddings_dir: Path = Path("precomputed_features")
    output_dir: Path = Path("umap_plots/single")
    split: Literal["train", "val", "test"] = "val"
    max_samples: int = MAX_SAMPLES
    max_per_class: int = MAX_PER_CLASS
    seed: int = 42
    dpi: int = 200
    n_neighbors: int = 15
    min_dist: float = 0.1


def _model_token(config: PlotUmapConfig) -> str:
    return _serialize_model_name(config.model)


def _dataset_token(config: PlotUmapConfig) -> str:
    return config.data.dataset


def find_h5(embeddings_dir: Path, model_tok: str, dataset_tok: str, split: str) -> Path | None:
    pattern = str(embeddings_dir / f"agg_*_{model_tok}_{dataset_tok}_{split}_*.h5")
    hits = _glob.glob(pattern)
    return Path(hits[0]) if hits else None


def load_and_subsample(h5_path: Path, max_samples: int, max_per_class: int, seed: int):
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:].astype(np.float32)
        labels = f["labels"][:]

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
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed
    )
    return reducer.fit_transform(features)


def colormap_for(n: int):
    name = "tab10" if n <= 10 else "tab20" if n <= 20 else "turbo"
    return plt.get_cmap(name, n)


def plot_umap(xy: np.ndarray, labels: np.ndarray, title: str, output_path: Path, dpi: int) -> None:
    n_cls = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=labels,
        cmap=colormap_for(n_cls),
        vmin=0, vmax=max(n_cls - 1, 1),
        s=6, alpha=0.6, linewidths=0,
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


def main(config: PlotUmapConfig) -> None:
    if config.model is None or config.data is None:
        raise ValueError("Both --model and --data config blocks must be provided.")

    model_tok = _model_token(config)
    dataset_tok = _dataset_token(config)

    print(f"Model token  : {model_tok}")
    print(f"Dataset token: {dataset_tok}")

    h5 = find_h5(config.embeddings_dir, model_tok, dataset_tok, config.split)
    if h5 is None:
        raise FileNotFoundError(
            f"No HDF5 found in {config.embeddings_dir} matching "
            f"model='{model_tok}', dataset='{dataset_tok}', split='{config.split}'.\n"
            "Run extract_features.py first to produce the embeddings."
        )

    print(f"Found H5: {h5}")
    print("Loading embeddings...")
    features, labels = load_and_subsample(
        h5, config.max_samples, config.max_per_class, config.seed
    )
    n_cls = len(np.unique(labels))
    print(f"  {len(labels):,} samples, {n_cls} classes")

    print("Fitting UMAP...")
    xy = fit_umap(features, config.n_neighbors, config.min_dist, config.seed)

    encoder = config.model.encoder
    model_label = getattr(encoder, "type", model_tok)
    title = (
        f"{model_label} / {dataset_tok}  ({config.split})\n"
        f"n={len(labels):,}  classes={n_cls}"
    )
    out = config.output_dir / f"{model_tok}_{dataset_tok}_{config.split}.png"
    plot_umap(xy, labels, title, out, config.dpi)


if __name__ == "__main__":
    main(parse_cli(PlotUmapConfig))