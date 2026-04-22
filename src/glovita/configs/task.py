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
    metrics: List[str] = ["acc", "f1"]
    # "stepwise": compute per batch, aggregate; "epochwise": accumulate then compute once
    metric_computation_mode: Literal["stepwise", "epochwise"] = "epochwise"
    # Whether to log confusion matrices / scatter plots to the active experiment logger
    result_plot: Optional[Literal["val", "all"]] = None
    # Label smoothing for cross-entropy (0 = disabled)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    # Mixup augmentation
    mixup: bool = False
    mixup_alpha: float = Field(default=0.2, gt=0.0)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        valid = {"acc", "balanced_acc", "f1", "f1_per_class", "pr", "top5acc", "auroc", "ap", "mse", "mae"}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}. Valid: {valid}")
        return v
