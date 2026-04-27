from __future__ import annotations

from pydantic import BaseModel, Field


class DataloadingConfig(BaseModel):
    """All PyTorch DataLoader settings used by train and inference runtimes.

    The runtime intentionally derives a few values at use time instead of
    mutating config state during validation:
    - `effective_eval_batch_size`: falls back to `batch_size`
    - `effective_persistent_workers`: disabled automatically for `num_workers=0`
    - `effective_prefetch_factor`: disabled automatically for `num_workers=0`
    """

    batch_size: int = Field(default=32, ge=1, description="Training batch size.")
    eval_batch_size: int | None = Field(default=None, ge=1, description="Validation/test batch size. Falls back to batch_size when unset.")
    num_workers: int = Field(default=12, ge=0, description="Number of DataLoader worker processes.")
    pin_memory: bool = Field(default=True, description="Enable pinned host memory for faster GPU transfers.")
    persistent_workers: bool = Field(default=True, description="Keep workers alive across epochs when num_workers > 0.")
    prefetch_factor: int | None = Field(default=2, ge=1, description="Number of prefetched batches per worker. Ignored when num_workers=0.")
    timeout: float = Field(default=0.0, ge=0.0, description="DataLoader worker timeout in seconds.")
    drop_last_train: bool = Field(default=False, description="Drop the last incomplete training batch.")
    drop_last_eval: bool = Field(default=False, description="Drop the last incomplete validation/test batch.")
    shuffle_train: bool = Field(default=True, description="Shuffle the training dataset each epoch.")
    shuffle_eval: bool = Field(default=False, description="Shuffle validation/test data. Usually left disabled.")
    use_worker_init_fn: bool = Field(default=True, description="Use the repo worker seeding function for reproducible workers.")

    @property
    def effective_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size

    @property
    def effective_persistent_workers(self) -> bool:
        return self.num_workers > 0 and self.persistent_workers

    @property
    def effective_prefetch_factor(self) -> int | None:
        if self.num_workers == 0:
            return None
        return self.prefetch_factor
