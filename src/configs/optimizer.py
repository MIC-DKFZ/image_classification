from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """Optimizer and learning-rate scheduler settings."""

    name: Literal["SGD", "Adam", "AdamW", "RMSprop", "Madgrad"] = "AdamW"
    lr: float = Field(default=2e-5, gt=0.0)
    weight_decay: float = Field(default=0.05, ge=0.0)
    # SGD-only: use Nesterov momentum
    nesterov: bool = False
    # BERT-style layer-wise LR decay for ViTs (e.g. 0.75). None = disabled
    layer_wise_lr_decay: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    # Leave bias and norm parameters undecayed (Bag of Tricks, arXiv:1812.01187)
    undecay_norm: bool = False

    # --- Scheduler ---
    scheduler: Optional[Literal["CosineAnneal", "Step", "MultiStep"]] = "CosineAnneal"
    # Linear warmup epochs at the start of training (0 = no warmup)
    warmup_epochs: int = Field(default=10, ge=0)
