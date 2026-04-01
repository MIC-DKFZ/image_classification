#!/usr/bin/env python
"""Inference / evaluation entry point.

Loads one or more checkpoints from a training run, runs inference on the test
set, and optionally ensembles logits (sum-before-softmax).

Usage
-----
Single fold:

    python infer.py \
        --exp_dir ./experiments/imagenet/my_run/0 \
        --data.dataset imagenet --data.data_root_dir /data/ILSVRC

Ensemble over all fold checkpoints (scans sub-directories automatically):

    python infer.py --exp_dir ./experiments/imagenet/my_run
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
import tyro
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from datasets.factory import build_dataloaders
from src.configs.data import DataConfig


class InferConfig(BaseModel):
    """Inference configuration."""

    # Directory that contains checkpoints/ or fold sub-directories
    exp_dir: Path = Path("./experiments")
    data: DataConfig
    metrics: List[str] = Field(default_factory=lambda: ["acc", "f1"])
    # Evaluate a specific fold only (None = scan all folds)
    fold: Optional[str] = None
    # Write predictions + labels to this file
    pred_output: Optional[Path] = None


def _collect_checkpoints(exp_dir: Path, fold: Optional[str]) -> List[Path]:
    if fold is not None:
        candidates = list((exp_dir / fold / "checkpoints").glob("last.pt"))
    else:
        candidates = list(exp_dir.glob("*/checkpoints/last.pt"))
        if not candidates:
            candidates = list((exp_dir / "checkpoints").glob("last.pt"))
    if not candidates:
        raise FileNotFoundError(f"No 'last.pt' checkpoints found under {exp_dir}")
    return sorted(candidates)


def _load_model(ckpt_path: Path) -> torch.nn.Module:
    """Re-create the model from the saved config.json and load checkpoint weights."""
    run_dir = ckpt_path.parent.parent  # checkpoints/ -> run dir
    config_file = run_dir / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_file}. "
            "Make sure the checkpoint was created by the new train.py."
        )

    from src.configs.root import RootConfig
    from models.peft.registry import apply_peft

    config = RootConfig.model_validate_json(config_file.read_text())

    from train import _build_model
    model = _build_model(config)
    model = apply_peft(model, config.peft)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(config: InferConfig) -> None:
    ckpt_paths = _collect_checkpoints(config.exp_dir, config.fold)
    print(f"Found {len(ckpt_paths)} checkpoint(s).")

    _, _, test_loader = build_dataloaders(config.data)

    all_logits: List[torch.Tensor] = []
    all_labels: Optional[torch.Tensor] = None

    for ckpt_path in ckpt_paths:
        print(f"  Loading {ckpt_path}")
        model = _load_model(ckpt_path)

        batch_logits, batch_labels = [], []
        for x, y in test_loader:
            batch_logits.append(model(x))
            batch_labels.append(y)

        all_logits.append(torch.cat(batch_logits))
        if all_labels is None:
            all_labels = torch.cat(batch_labels)

    assert all_labels is not None

    # Ensemble: sum logits, then argmax
    summed = torch.sum(torch.stack(all_logits), dim=0)
    preds = torch.argmax(summed, dim=1)

    # Compute requested metrics
    from torchmetrics import MetricCollection, Accuracy, F1Score

    metrics_dict = {}
    if "acc" in config.metrics:
        metrics_dict["Accuracy"] = Accuracy(
            task="multiclass", num_classes=config.data.num_classes
        )
    if "f1" in config.metrics:
        metrics_dict["F1"] = F1Score(
            task="multiclass", num_classes=config.data.num_classes, average="macro"
        )

    collection = MetricCollection(metrics_dict)
    results = collection(preds, all_labels)

    print("\nTest results:")
    for k, v in results.items():
        print(f"  {k}: {v.item():.4f}")

    if config.pred_output is not None:
        config.pred_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"preds": preds, "labels": all_labels}, config.pred_output)
        print(f"Predictions saved to {config.pred_output}")


if __name__ == "__main__":
    run_inference(tyro.cli(InferConfig))
