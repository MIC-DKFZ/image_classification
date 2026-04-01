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

import os
import sys
from pathlib import Path

# Ensure the project root is importable regardless of how the script is invoked
sys.path.insert(0, str(Path(__file__).parent))

import torch
import tyro
import wandb

from src.configs.root import RootConfig
from src.training.trainer import Trainer
from src.training.optimizers import build_optimizer
from src.training.schedulers import build_scheduler
from datasets.factory import build_dataloaders
from models.factory import build_model
from models.peft.registry import apply_peft


def _build_model(config: RootConfig) -> torch.nn.Module:
    """Instantiate the composed encoder + head model from config."""
    return build_model(config.model, output_dim=getattr(config.data, "num_classes", 1))


def run_fold(config: RootConfig, fold: int) -> None:
    """Train a single cross-validation fold (or the single no-CV run)."""
    fold_str = str(fold) if config.training.cv_folds > 1 else "0"
    log_dir = config.run_log_dir / fold_str
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    wandb_kwargs = config.wandb.model_dump(exclude_none=True)
    offline = wandb_kwargs.pop("offline", False)
    if offline:
        wandb_kwargs["mode"] = "offline"
    wandb_kwargs["dir"] = str(log_dir)
    if config.training.cv_folds > 1:
        # Tag each fold run distinctly within the same group
        wandb_kwargs.setdefault("tags", [])
        wandb_kwargs["tags"] = list(wandb_kwargs["tags"] or []) + [f"fold_{fold}"]
    wandb.init(**wandb_kwargs)

    # Log hyperparams as a flat dict
    wandb.config.update(
        {
            "model": config.model.model_dump(),
            "peft": config.peft.model_dump(),
            "data": {k: str(v) if isinstance(v, Path) else v
                     for k, v in config.data.model_dump().items()},
            "optimizer": config.optimizer.model_dump(),
            "training": {k: str(v) if isinstance(v, Path) else v
                         for k, v in config.training.model_dump().items()},
        }
    )

    # --- Data ---
    fold_data = config.data.model_copy(update={"fold": fold_str})
    train_loader, val_loader, _ = build_dataloaders(fold_data)

    # --- Model + PEFT ---
    model = _build_model(config)
    model = apply_peft(model, config.peft)

    # --- Optimizer + Scheduler ---
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.optimizer, config.training.epochs)

    # Save config snapshot for inference / reproducibility
    config_path = log_dir / "config.json"
    config_path.write_text(config.model_dump_json(indent=2))

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

    for fold in range(config.training.cv_folds):
        run_fold(config, fold)


if __name__ == "__main__":
    main(tyro.cli(RootConfig))
