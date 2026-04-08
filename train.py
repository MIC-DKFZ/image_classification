#!/usr/bin/env python
"""Training entry point.

Usage examples
--------------
Basic run (tyro generates the full CLI from RootConfig):

    python train.py \
        --data.dataset imagenet --data.data_root_dir /data/ILSVRC \
        --model.encoder.encoder_type timm --model.encoder.type vit_base_patch16_224 \
        --model.head.head_type classification \
        --peft.method lora --peft.lora_rank 16 \
        --training.epochs 20 --optimizer.lr 2e-5

Select subcommands (tyro's shorthand for discriminated unions):

    python train.py \
        data:imagenet-config --data.data_root_dir /data/ILSVRC \
        --model.encoder.encoder_type timm --model.encoder.type vit_base_patch16_224 \
        --model.head.head_type classification \
        peft:lora-config --peft.lora_rank 16

Multi-GPU with Accelerate (launch via accelerate CLI):

    accelerate launch --num_processes 4 train.py ...

Cross-validation (5 folds):

    python train.py ... --training.cv_folds 5
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import torch
import wandb

from glovita.configs.cli import parse_root_cli
from glovita.configs.root import RootConfig
from glovita.training.trainer import Trainer
from glovita.training.optimizers import build_optimizer
from glovita.training.schedulers import build_scheduler
from glovita.datasets.factory import build_dataloaders, resolve_augmentation_config
from glovita.models.factory import build_model
from glovita.models.peft.registry import apply_peft
from glovita.models.preprocessing import resolve_encoder_preprocessing_defaults


def _build_model(config: RootConfig) -> torch.nn.Module:
    """Instantiate the composed encoder + head model from config."""
    return build_model(config.model, output_dim=getattr(config.data, "num_classes", 1))


def _serialize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _collect_runtime_metadata(
    config: RootConfig,
    fold_data,
    encoder_preprocessing: dict,
    effective_wandb: dict,
) -> dict:
    return {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "timm": _package_version("timm"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "wandb": _package_version("wandb"),
            "tyro": _package_version("tyro"),
            "pydantic": _package_version("pydantic"),
        },
        "cuda": {
            "is_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "resolved": {
            "encoder_preprocessing": _serialize(encoder_preprocessing),
            "augmentation": _serialize(resolve_augmentation_config(fold_data, encoder_preprocessing)),
            "data": _serialize(fold_data.model_dump()),
            "model": _serialize(config.model.model_dump()),
            "peft": _serialize(config.peft.model_dump()),
            "optimizer": _serialize(config.optimizer.model_dump()),
            "training": _serialize(config.training.model_dump()),
            "task": _serialize(config.task.model_dump()),
            "wandb": _serialize(effective_wandb),
            "add_log": _serialize(config.add_log),
        },
    }


def run_fold(config: RootConfig, fold: str) -> None:
    """Train a single cross-validation fold (or the single no-CV run)."""
    wandb_kwargs = config.resolve_wandb_kwargs()
    effective_group = wandb_kwargs["group"]
    log_dir = config.get_run_log_dir(effective_group) / fold
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    offline = wandb_kwargs.pop("offline", False)
    if offline:
        wandb_kwargs["mode"] = "offline"
    wandb_kwargs["dir"] = str(log_dir)
    if config.training.cv_folds > 1 or config.data.fold is not None:
        # Tag each fold run distinctly within the same group
        wandb_kwargs.setdefault("tags", [])
        wandb_kwargs["tags"] = list(wandb_kwargs["tags"] or []) + [f"fold_{fold}"]
    wandb.init(**wandb_kwargs)

    # --- Data ---
    fold_data = config.data.model_copy(update={"fold": fold})
    encoder_preprocessing = resolve_encoder_preprocessing_defaults(config.model.encoder).as_kwargs()
    runtime_metadata = _collect_runtime_metadata(
        config,
        fold_data,
        encoder_preprocessing,
        effective_wandb=wandb_kwargs,
    )

    # Log full user config plus resolved runtime settings
    wandb.config.update(runtime_metadata["resolved"])
    wandb.config.update(
        {"runtime": _serialize({k: v for k, v in runtime_metadata.items() if k != "resolved"})},
        allow_val_change=True,
    )

    train_loader, val_loader, _ = build_dataloaders(
        fold_data,
        config.dataloading,
        encoder_preprocessing=encoder_preprocessing,
    )

    # --- Model + PEFT ---
    model = _build_model(config)
    model = apply_peft(model, config.peft)

    # --- Optimizer + Scheduler ---
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.optimizer, config.training.epochs)

    # Save config snapshot for inference / reproducibility
    config_path = log_dir / "config.json"
    config_path.write_text(config.model_dump_json(indent=2))
    resolved_config_path = log_dir / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(runtime_metadata["resolved"], indent=2))
    runtime_path = log_dir / "runtime_info.json"
    runtime_path.write_text(json.dumps(runtime_metadata, indent=2))

    # --- Train ---
    trainer = Trainer(config.training, config.task, config.data, log_dir)
    trainer.fit(model, optimizer, scheduler, train_loader, val_loader)

    wandb.finish()


def main(config: RootConfig) -> None:
    # Reproducibility
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Increase W&B service wait time for slow cluster nodes
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

    if config.data.fold is not None:
        run_fold(config, config.data.fold)
        return

    for fold in range(config.training.cv_folds):
        run_fold(config, str(fold))


if __name__ == "__main__":
    main(parse_root_cli(RootConfig))
