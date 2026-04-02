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

    batch_size: int = Field(default=32, ge=1)
    eval_batch_size: int | None = Field(default=None, ge=1)
    num_workers: int = Field(default=12, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = Field(default=2, ge=1)
    timeout: float = Field(default=0.0, ge=0.0)
    drop_last_train: bool = False
    drop_last_eval: bool = False
    shuffle_train: bool = True
    shuffle_eval: bool = False
    use_worker_init_fn: bool = True

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
