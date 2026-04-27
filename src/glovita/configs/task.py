from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TaskConfig(BaseModel):
    """Task definition: what to predict and how to measure it.

    These settings are shared across all datasets for a given experiment run.
    Dataset-specific values (num_classes, task type) live in DataConfig.
    """

    # Which metrics to track. Options: acc, balanced_acc, f1, f1_per_class,
    # pr (precision+recall), top5acc, auroc, ap, mse, mae
    metrics: List[str] = Field(default_factory=lambda: ["acc", "f1"], description="Metric names to compute during validation and inference.")
    # "stepwise": compute per batch, aggregate; "epochwise": accumulate then compute once
    metric_computation_mode: Literal["stepwise", "epochwise"] = Field(default="epochwise", description="Whether metrics are computed per batch or accumulated across the full epoch.")
    # Whether to log confusion matrices / scatter plots to the active experiment logger
    result_plot: Optional[Literal["val", "all"]] = Field(default=None, description="Log confusion matrices or regression scatter plots for validation only or for all splits.")
    # Label smoothing for cross-entropy (0 = disabled)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    # Mixup augmentation
    mixup: bool = Field(default=False, description="Enable mixup augmentation in the training loop.")
    mixup_alpha: float = Field(default=0.2, gt=0.0, description="Beta distribution alpha parameter used by mixup.")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        valid = {"acc", "balanced_acc", "f1", "f1_per_class", "pr", "top5acc", "auroc", "ap", "mse", "mae"}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}. Valid: {valid}")
        return v
