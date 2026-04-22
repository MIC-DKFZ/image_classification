from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TrainingConfig(BaseModel):
    """Controls the training loop and hardware settings."""

    epochs: int = 20
    # Accelerate precision: "no" = fp32, "fp16", "bf16"
    precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 1
    gradient_clip_val: Optional[float] = None
    seed: Optional[int] = None
    # torch.compile the model before training
    compile: bool = False
    enable_checkpointing: bool = True
    # Logged validation metric key used to decide whether to update `best.pt`.
    # Examples: "Accuracy", "F1", "Balanced_Accuracy", "MSE", "MAE".
    # If unset, the first logged validation metric is used.
    best_checkpoint_metric: str | None = None
    # Throttle progress bar output for LSF/SLURM environments
    cluster_progress_bar: bool = False
    # Cross-validation: 1 = no CV, k>1 = k-fold CV
    cv_folds: int = Field(default=1, ge=1)
    # Number of sanity validation steps before training (0 = disabled)
    num_sanity_val_steps: int = 0

    @field_validator("gradient_clip_val")
    @classmethod
    def clip_val_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("gradient_clip_val must be positive")
        return v
