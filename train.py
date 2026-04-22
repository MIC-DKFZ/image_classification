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

from glovita.configs.cli import parse_root_cli
from glovita.configs.root import RootConfig
from glovita.datasets.factory import build_dataloaders, resolve_augmentation_config
from glovita.logging import build_logger
from glovita.models.factory import build_model
from glovita.models.peft.registry import apply_peft
from glovita.models.preprocessing import resolve_encoder_preprocessing_defaults
from glovita.training.optimizers import build_optimizer
from glovita.training.schedulers import build_scheduler
from glovita.training.trainer import Trainer


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
    effective_logger: dict,
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
            "mlflow": _package_version("mlflow"),
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
            "logger": _serialize(effective_logger),
            "add_log": _serialize(config.add_log),
        },
    }


def _resolve_effective_logger_config(config: RootConfig, group: str) -> dict:
    logger_cfg = config.logger.model_dump(exclude_none=True)
    backend = logger_cfg.get("backend")
    if backend == "wandb":
        logger_cfg.setdefault("project", getattr(config.logger, "project", None) or config.default_logger_experiment_name)
        logger_cfg.setdefault("group", getattr(config.logger, "group", None) or group)
    elif backend == "mlflow":
        logger_cfg.setdefault("tracking_uri", getattr(config.logger, "tracking_uri", None) or config.default_mlflow_tracking_uri)
        logger_cfg.setdefault(
            "experiment_name",
            getattr(config.logger, "experiment_name", None) or config.default_logger_experiment_name,
        )
        logger_cfg.setdefault("group", getattr(config.logger, "group", None) or group)
    return logger_cfg


def run_fold(config: RootConfig, fold: str, group: str) -> None:
    """Train a single cross-validation fold (or the single no-CV run)."""
    effective_group = group
    log_dir = config.get_run_log_dir(effective_group) / fold
    log_dir.mkdir(parents=True, exist_ok=True)
    extra_tags = [f"fold_{fold}"] if (config.training.cv_folds > 1 or config.data.fold is not None) else []
    logger = build_logger(
        config.logger,
        log_dir=log_dir,
        default_experiment_name=config.default_logger_experiment_name,
        default_tracking_uri=config.default_mlflow_tracking_uri,
        group=effective_group,
        extra_tags=extra_tags,
    )

    # --- Data ---
    fold_data = config.data.model_copy(update={"fold": fold})
    encoder_preprocessing = resolve_encoder_preprocessing_defaults(config.model.encoder).as_kwargs()
    runtime_metadata = _collect_runtime_metadata(
        config,
        fold_data,
        encoder_preprocessing,
        effective_logger=_resolve_effective_logger_config(config, effective_group),
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
    logger.log_config(runtime_metadata["resolved"])
    logger.log_config({"runtime": _serialize({k: v for k, v in runtime_metadata.items() if k != "resolved"})})

    # --- Train ---
    try:
        trainer = Trainer(config.training, config.task, fold_data, log_dir, logger)
        trainer.fit(model, optimizer, scheduler, train_loader, val_loader)
    finally:
        logger.finish()


def main(config: RootConfig) -> None:
    # Reproducibility
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if getattr(config.logger, "backend", None) == "wandb":
        # Increase W&B service wait time for slow cluster nodes.
        os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

    # Resolve the run group once so all fold runs share the same grouping key.
    explicit_group = getattr(config.logger, "group", None)
    group = explicit_group or config.generate_run_group()

    if config.data.fold is not None:
        run_fold(config, config.data.fold, group=group)
        return

    for fold in range(config.training.cv_folds):
        run_fold(config, str(fold), group=group)
        # Release model/optimizer memory between folds so large models don't
        # accumulate across all folds in the same process.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(parse_root_cli(RootConfig))
