"""Learning-rate schedulers.

Only the schedulers actually used in the codebase are kept here.
The sawtooth / double-warmstart scheduler has been removed.
"""
from __future__ import annotations

import math
import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler

from src.configs.optimizer import OptimizerConfig


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimizerConfig,
    total_epochs: int,
) -> _LRScheduler | None:
    """Return a scheduler for *optimizer*, or ``None`` if no scheduler is configured."""
    if not config.scheduler:
        return None

    if config.scheduler == "CosineAnneal":
        if config.warmup_epochs > 0:
            return CosineAnnealingLR_Warmstart(
                optimizer,
                T_max=total_epochs,
                warmstart=config.warmup_epochs,
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs
        )

    if config.scheduler == "Step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_epochs // 4), gamma=0.1
        )

    if config.scheduler == "MultiStep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[total_epochs // 2, total_epochs * 3 // 4]
        )

    raise ValueError(f"Unknown scheduler: {config.scheduler!r}")


class CosineAnnealingLR_Warmstart(_LRScheduler):
    """CosineAnnealingLR with a linear warmup phase.

    During the first ``warmstart`` epochs the LR is linearly increased from
    zero to ``base_lr``.  After that, standard cosine annealing runs for the
    remaining ``T_max - warmstart`` epochs.

    Reference: https://arxiv.org/pdf/1706.02677.pdf
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        warmstart: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmstart = warmstart
        self.T_max = T_max - warmstart  # effective cosine period
        self.eta_min = eta_min
        self._T = 0  # internal cosine counter

        try:
            super().__init__(optimizer, last_epoch=last_epoch, verbose=False)
        except TypeError:
            super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate use `get_last_lr()`.",
                UserWarning,
            )

        # Linear warmup
        if self.last_epoch < self.warmstart:
            factor = (self.last_epoch + 1) / (self.warmstart + 1)
            return [base_lr * factor for base_lr in self.base_lrs]

        # First step after warmup: return base_lrs unchanged
        if self._T == 0:
            self._T += 1
            return list(self.base_lrs)

        # Cosine annealing
        if (self._T - 1 - self.T_max) % (2 * self.T_max) == 0:
            lrs = [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
            self._T += 1
            return lrs

        lrs = [
            (1 + math.cos(math.pi * self._T / self.T_max))
            / (1 + math.cos(math.pi * (self._T - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
        self._T += 1
        return lrs
