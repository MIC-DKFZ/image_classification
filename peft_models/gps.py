# gps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import numpy as np
from contextlib import nullcontext

_logger = logging.getLogger("train")


@dataclass
class GPSConfig:
    # NOTE: this is NOT a percent; it's top-k per "row/cell-group" like your original code
    topk_per_row: int = 1
    calib_batches: int = 1  # number of batches to probe gradients on
    keep_trainable_name_substrings: Tuple[str, ...] = ("head", "classifier", "cls_head")
    # always freeze these (name contains)
    always_freeze_name_substrings: Tuple[str, ...] = ("norm", "pos_embed", "cls_token")


class GPS:
    """
    Gradient-based Parameter Selection (GPS) mixin for LightningModule.

    Expects:
      self.model      : nn.Module
      self.criterion  : callable(logits, targets) -> loss
      forward() returns logits used by criterion
    """

    def __init__(self, gps_percent: int = 1, gps_calib_batches: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gps_cfg = GPSConfig(topk_per_row=gps_percent, calib_batches=gps_calib_batches)
        self._gps_done = False
        self._gps_masks: Dict[str, torch.Tensor] = {}
        self._gps_hook_handles = []

    # ---------------- calibration loader ----------------
    def _build_calibration_loader(self) -> DataLoader:
        base_loader = self.trainer.datamodule.train_dataloader()
        dataset = base_loader.dataset
        bs = getattr(base_loader, "batch_size", 32) or 32

        # sample up to calib_batches * bs examples
        k = min(len(dataset), bs * max(1, int(self.gps_cfg.calib_batches)))
        idx = torch.randperm(len(dataset))[:k].tolist()

        return DataLoader(
            Subset(dataset, idx),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=getattr(base_loader, "pin_memory", True),
            drop_last=False,
        )

    # ---------------- Lightning hook ----------------
    def on_fit_start(self):
        if self._gps_done:
            return

        assert hasattr(self, "model"), "GPS expects self.model (nn.Module)"
        assert hasattr(self, "criterion"), "GPS expects self.criterion (loss fn)"

        calib_loader = self._build_calibration_loader()

        # Populate .grad by doing forward+backward (no optimizer.step)
        calculate_gradient(
            model=self,
            loader=calib_loader,
            loss_fn=self.criterion,
            amp_autocast=_use_amp_from_trainer(self.trainer),
            max_batches=self.gps_cfg.calib_batches,
        )

        # Build per-parameter train masks (1=train, 0=frozen)
        masks = build_train_masks_percell_topk(
            model=self.model,
            topk_per_row=int(self.gps_cfg.topk_per_row),
            always_freeze=self.gps_cfg.always_freeze_name_substrings,
            always_trainable=self.gps_cfg.keep_trainable_name_substrings,
        )
        self._gps_masks = masks

        # Apply masks: register grad hooks, and fully-freeze tensors if mask is all-zero
        apply_gradient_masks_(self.model, masks, store_handles_list=self._gps_hook_handles)

        # Clear probe grads
        self.model.zero_grad(set_to_none=True)

        # Rebuild optimizers so they can drop parameters that became fully requires_grad=False
        # (per-element masking still needs hooks; this only helps for fully frozen tensors)
        self.trainer.strategy.setup_optimizers(self.trainer)

        self._gps_done = True
        if hasattr(self, "print"):
            self.print(f"[GPS] Applied per-cell top-{self.gps_cfg.topk_per_row} gradient masking; optimizers reinitialized.")


# ---------------- helpers ----------------

def _use_amp_from_trainer(trainer) -> bool:
    prec = str(getattr(trainer, "precision", "")).lower()
    # lightning can be "16-mixed", "bf16-mixed", etc.
    return ("16" in prec) or ("bf16" in prec)


def calculate_gradient(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    amp_autocast: bool = False,
    max_batches: int = 1,
):
    """
    Run forward+backward to populate .grad on params.
    No optimizer.step(). No distributed tricks.
    """
    model.train()
    model.zero_grad(set_to_none=True)

    device = next(model.parameters()).device
    use_cuda_amp = amp_autocast and (device.type == "cuda")

    if use_cuda_amp:
        # torch.amp.autocast is the modern API
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    for bidx, (inputs, targets) in enumerate(loader):
        if bidx >= max(1, int(max_batches)):
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_ctx:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        loss.backward()


def build_train_masks_percell_topk(
    model: torch.nn.Module,
    topk_per_row: int,
    always_freeze: Tuple[str, ...],
    always_trainable: Tuple[str, ...],
) -> Dict[str, torch.Tensor]:
    """
    Returns dict[name] = train_mask tensor on CPU (float32), same shape as param:
      1.0 => trainable
      0.0 => frozen

    "Per-cell topk" behavior:
      - For tensors with ndim >= 2: reshape to [rows, cols] where rows = dim0 and cols = prod(rest),
        then for each row keep top-k |grad| entries (train_mask=1).
      - For tensors with ndim == 1: keep none by default unless always_trainable matches.
        (Biases are handled by name rule below.)
    """
    masks: Dict[str, torch.Tensor] = {}
    k = max(0, int(topk_per_row))

    for name, p in model.named_parameters():
        # Name-based rules
        if any(s in name for s in always_freeze):
            train_mask = torch.zeros_like(p.detach(), device="cpu", dtype=torch.float32)
            masks[name] = train_mask
            continue

        # keep heads / classifiers / etc trainable
        if any(s in name for s in always_trainable) or ("bias" in name) or ("gamma" in name):
            train_mask = torch.ones_like(p.detach(), device="cpu", dtype=torch.float32)
            masks[name] = train_mask
            continue

        # No grad produced => freeze
        if p.grad is None:
            train_mask = torch.zeros_like(p.detach(), device="cpu", dtype=torch.float32)
            masks[name] = train_mask
            continue

        g = p.grad.detach().float().cpu()
        if g.numel() == 0:
            train_mask = torch.zeros_like(p.detach(), device="cpu", dtype=torch.float32)
            masks[name] = train_mask
            continue

        if g.ndim < 2:
            # 1D parameters: default freeze (unless name rules above made it trainable)
            train_mask = torch.zeros_like(g, dtype=torch.float32)
            masks[name] = train_mask
            continue

        rows = g.shape[0]
        cols = int(np.prod(g.shape[1:]))

        g2 = g.reshape(rows, cols).abs()

        if k <= 0:
            m2 = torch.zeros_like(g2, dtype=torch.float32)
        elif k >= cols:
            m2 = torch.ones_like(g2, dtype=torch.float32)
        else:
            # topk indices per row
            # torch.topk is fine here (small calib batch)
            _, idx = torch.topk(g2, k=k, dim=1, largest=True, sorted=False)
            m2 = torch.zeros_like(g2, dtype=torch.float32)
            m2.scatter_(1, idx, 1.0)

        train_mask = m2.reshape(g.shape)
        # ensure same shape as param (should be)
        if train_mask.shape != p.shape:
            # fallback: broadcast-safe freeze (shouldn't happen, but don't crash)
            _logger.warning(f"[GPS] Mask shape mismatch for {name}: {train_mask.shape} vs {tuple(p.shape)}. Freezing tensor.")
            train_mask = torch.zeros_like(p.detach(), device="cpu", dtype=torch.float32)

        masks[name] = train_mask

    # Optional: print stats
    _print_mask_stats(masks)
    return masks


def apply_gradient_masks_(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    store_handles_list: Optional[list] = None,
):
    """
    Applies per-element masks by registering backward hooks:
        grad <- grad * train_mask

    Also sets requires_grad=False for tensors that are fully frozen (mask sum == 0),
    so optimizers can ignore them.
    """
    name_to_param = dict(model.named_parameters())

    for name, train_mask_cpu in masks.items():
        p = name_to_param.get(name, None)
        if p is None:
            continue

        # if completely frozen, flip requires_grad off (saves optimizer work)
        if float(train_mask_cpu.sum().item()) == 0.0:
            p.requires_grad_(False)
            continue
        else:
            p.requires_grad_(True)

        # Register hook for per-element freezing
        # Move mask to param device once (still float32); cast at runtime to grad dtype
        train_mask = train_mask_cpu.to(device=p.device, non_blocking=True)

        def _hook(grad, m=train_mask):
            if grad is None:
                return None
            return grad * m.to(dtype=grad.dtype)

        h = p.register_hook(_hook)
        if store_handles_list is not None:
            store_handles_list.append(h)


def _print_mask_stats(masks: Dict[str, torch.Tensor]):
    total = 0
    trainable = 0
    for n, m in masks.items():
        t = m.numel()
        tr = int(m.sum().item())
        total += t
        trainable += tr
    pct = (100.0 * trainable / total) if total else 0.0
    print("---------------------------------------------------------------")
    print(f"[GPS] Trainable elements / Total elements: {trainable} / {total} ({pct:.4f}%)")
    print("---------------------------------------------------------------")
