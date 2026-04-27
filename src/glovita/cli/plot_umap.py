from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field

from glovita.configs.cli import parse_cli

try:
    import umap as umap_module
except ImportError as exc:
    raise ImportError("Install umap-learn: pip install umap-learn") from exc


MAX_SAMPLES = 5000
MAX_PER_CLASS = 200


class PlotUmapConfig(BaseModel):
    """UMAP plot configuration for an extracted feature file."""

    h5_path: Path = Field(description="HDF5 feature file produced by glovita_extract_features.")
    title: str | None = Field(default=None, description="Optional plot title. Defaults to the HDF5 file stem.")
    output_path: Path | None = Field(default=None, description="Optional output image path. Defaults to umap_plots/single/<h5_stem>.png.")
    max_samples: int = Field(default=MAX_SAMPLES, ge=1, description="Maximum total number of samples to plot after subsampling.")
    max_per_class: int = Field(default=MAX_PER_CLASS, ge=1, description="Maximum number of samples kept per class before the global max_samples cap is applied.")
    seed: int = Field(default=42, description="Random seed used for subsampling and UMAP initialization.")
    dpi: int = Field(default=200, ge=1, description="Output image resolution in dots per inch.")
    n_neighbors: int = Field(default=15, ge=2, description="UMAP n_neighbors parameter controlling local neighborhood size.")
    min_dist: float = Field(default=0.1, ge=0.0, description="UMAP min_dist parameter controlling cluster compactness.")


def load_and_subsample(h5_path: Path, max_samples: int, max_per_class: int, seed: int):
    with h5py.File(h5_path, "r") as f:
        if "features" not in f or "labels" not in f:
            raise KeyError(f"{h5_path} must contain 'features' and 'labels' datasets.")
        features = f["features"][:].astype(np.float32)
        labels = f["labels"][:]

    if features.ndim != 2:
        raise ValueError(
            f"{h5_path} contains features with shape {features.shape}. "
            "glovita_plot_umap expects instance-level features with shape (N, D)."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"{h5_path} contains labels with shape {labels.shape}. "
            "glovita_plot_umap expects labels with shape (N,)."
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


def run_plot(config: PlotUmapConfig) -> None:
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


def main(config: PlotUmapConfig | None = None) -> None:
    if config is None:
        config = parse_cli(PlotUmapConfig)
    run_plot(config)


if __name__ == "__main__":
    main()
