from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from pydantic import BaseModel, Field

from glovita.configs.cli import parse_cli
from glovita.configs.data import DataConfig
from glovita.configs.dataloading import DataloadingConfig
from glovita.datasets.factory import build_dataloaders
from glovita.models.preprocessing import resolve_encoder_preprocessing_defaults


class InferConfig(BaseModel):
    exp_dir: Path = Path("./experiments")
    data: DataConfig
    dataloading: DataloadingConfig = Field(default_factory=DataloadingConfig)
    metrics: List[str] = Field(default_factory=lambda: ["acc", "f1"])
    fold: Optional[str] = None
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


def _load_run_config(ckpt_path: Path):
    run_dir = ckpt_path.parent.parent
    config_file = run_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found at {config_file}")
    from glovita.configs.root import RootConfig

    return RootConfig.model_validate_json(config_file.read_text())


def _load_model(ckpt_path: Path) -> torch.nn.Module:
    from glovita.models.factory import build_model
    from glovita.models.peft.registry import apply_peft

    run_config = _load_run_config(ckpt_path)
    output_dim = getattr(run_config.data, "num_classes", 1)
    model = build_model(run_config.model, output_dim=output_dim)
    model = apply_peft(model, run_config.peft)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(config: InferConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = _collect_checkpoints(config.exp_dir, config.fold)
    print(f"Found {len(ckpt_paths)} checkpoint(s).")

    reference_run_config = _load_run_config(ckpt_paths[0])
    encoder_preprocessing = resolve_encoder_preprocessing_defaults(
        reference_run_config.model.encoder
    ).as_kwargs()
    _, _, test_loader = build_dataloaders(
        config.data,
        config.dataloading,
        encoder_preprocessing=encoder_preprocessing,
    )

    task = reference_run_config.data.task
    subtask = reference_run_config.data.subtask
    num_classes = reference_run_config.data.num_classes

    all_logits: List[torch.Tensor] = []
    all_labels: Optional[torch.Tensor] = None

    for ckpt_path in ckpt_paths:
        print(f"  Loading {ckpt_path}")
        model = _load_model(ckpt_path)
        model.to(device)

        batch_logits, batch_labels = [], []
        for x, y in test_loader:
            batch_logits.append(model(x.to(device)).cpu())
            batch_labels.append(y)

        all_logits.append(torch.cat(batch_logits))
        if all_labels is None:
            all_labels = torch.cat(batch_labels)

    assert all_labels is not None

    summed = torch.sum(torch.stack(all_logits), dim=0)

    if task == "Regression":
        preds = summed.squeeze(-1)
    elif subtask == "multilabel":
        preds = (summed.sigmoid() > 0.5).long()
    else:
        preds = torch.argmax(summed, dim=1)

    from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError, MetricCollection

    metrics_dict = {}
    if task == "Regression":
        if "mse" in config.metrics:
            metrics_dict["MSE"] = MeanSquaredError()
        if "mae" in config.metrics:
            metrics_dict["MAE"] = MeanAbsoluteError()
    else:
        metric_task = subtask
        if "acc" in config.metrics:
            metrics_dict["Accuracy"] = Accuracy(
                task=metric_task, num_classes=num_classes, num_labels=num_classes
            )
        if "f1" in config.metrics:
            metrics_dict["F1"] = F1Score(
                task=metric_task,
                num_classes=num_classes,
                num_labels=num_classes,
                average="macro",
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


def main(config: InferConfig | None = None) -> None:
    if config is None:
        config = parse_cli(InferConfig)
    run_inference(config)


if __name__ == "__main__":
    main()

