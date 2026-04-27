from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """Optimizer and learning-rate scheduler settings."""

    name: Literal["SGD", "Adam", "AdamW", "RMSprop", "Madgrad", "Muon"] = Field(default="AdamW", description="Optimizer name.")
    lr: float = Field(default=2e-5, gt=0.0, description="Base learning rate.")
    weight_decay: float = Field(default=0.05, ge=0.0, description="Weight decay coefficient.")
    # SGD-only: use Nesterov momentum
    nesterov: bool = Field(default=False, description="Use Nesterov momentum with SGD.")
    # Muon-specific parameters (mirrors torch.optim.Muon)
    muon_momentum: float = Field(default=0.95, ge=0.0, description="Muon momentum parameter.")
    muon_ns_coefficients: tuple[float, float, float] = Field(default=(3.4445, -4.775, 2.0315), description="Muon Newton-Schulz coefficients.")
    muon_eps: float = Field(default=1e-7, gt=0.0, description="Muon epsilon value.")
    muon_ns_steps: int = Field(default=5, ge=1, description="Number of Newton-Schulz iterations in Muon.")
    muon_adjust_lr_fn: Optional[Literal["original", "match_rms_adamw"]] = Field(default=None, description="Optional Muon learning-rate adjustment mode.")
    # BERT-style layer-wise LR decay for ViTs (e.g. 0.75). None = disabled
    layer_wise_lr_decay: Optional[float] = Field(default=None, gt=0.0, lt=1.0, description="Per-layer learning-rate decay factor for transformer-style backbones.")
    # Leave bias and norm parameters undecayed (Bag of Tricks, arXiv:1812.01187)
    undecay_norm: bool = Field(default=False, description="Exclude normalization and bias parameters from weight decay.")

    # --- Scheduler ---
    scheduler: Optional[Literal["CosineAnneal", "Step", "MultiStep"]] = Field(default="CosineAnneal", description="Learning-rate scheduler name.")
    # Linear warmup epochs at the start of training (0 = no warmup)
    warmup_epochs: int = Field(default=10, ge=0, description="Number of linear warmup epochs before the main scheduler.")
