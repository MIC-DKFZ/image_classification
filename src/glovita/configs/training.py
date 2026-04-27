from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TrainingConfig(BaseModel):
    """Controls the training loop and hardware settings."""

    epochs: int = Field(default=20, ge=1, description="Number of training epochs.")
    # Accelerate precision: "no" = fp32, "fp16", "bf16"
    precision: Literal["no", "fp16", "bf16"] = Field(default="bf16", description="Mixed-precision mode used by Accelerate.")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Number of optimization steps to accumulate before each optimizer update.")
    gradient_clip_val: Optional[float] = Field(default=None, description="Clip global gradient norm to this value.")
    seed: Optional[int] = Field(default=None, description="Global random seed.")
    # torch.compile the model before training
    compile: bool = Field(default=False, description="Compile the model with torch.compile before training.")
    enable_checkpointing: bool = Field(default=True, description="Write last.pt and best.pt checkpoints during training.")
    # Logged validation metric key used to decide whether to update `best.pt`.
    # Examples: "Accuracy", "F1", "Balanced_Accuracy", "MSE", "MAE".
    # If unset, the first logged validation metric is used.
    best_checkpoint_metric: str | None = None
    # Throttle progress bar output for LSF/SLURM environments
    cluster_progress_bar: bool = Field(default=False, description="Use reduced progress-bar output for cluster environments.")
    # Cross-validation: 1 = no CV, k>1 = k-fold CV
    cv_folds: int = Field(default=1, ge=1)
    # Number of sanity validation steps before training (0 = disabled)
    num_sanity_val_steps: int = Field(default=0, ge=0, description="Number of validation batches to run before the first training epoch.")

    @field_validator("gradient_clip_val")
    @classmethod
    def clip_val_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("gradient_clip_val must be positive")
        return v
