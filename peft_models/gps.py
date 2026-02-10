# gps.py
from types import SimpleNamespace
from typing import Sequence
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import numpy as np
from timm.utils import *
from timm.loss import *
from contextlib import nullcontext
from torch.cuda.amp import autocast as amp_autocast_ctx
from torch.amp import autocast_mode
import time
import logging
from contextlib import suppress
_logger = logging.getLogger('train')


class GPS:
    """
    Add Gradient-based Parameter Selection (GPS) to a LightningModule.

    Inherit like:
        class LitModel(GPS, pl.LightningModule): ...

    Expects the child to define:
        self.model       : nn.Module
        self.criterion   : loss function
        self.hparams.model.fine_tuning == "gps" to enable
        self.hparams.model.gps.percent : int (e.g., 1, 2 or 3)
        optional:
            self.hparams.model.gps.calib_batches : int (default 1)
            self.hparams.model.gps.keep_heads    : list[str] (default ["head","classifier","cls_head"])
    """
    def __init__(self, gps_percent: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gps_percent = gps_percent

    _gps_done: bool = False
    def _build_calibration_loader(self) -> DataLoader:
        """
        Create a tiny throwaway loader from the training dataset so we don't
        consume the real train dataloader/sampler state.
        """
        base_loader = self.trainer.datamodule.train_dataloader()
        dataset = base_loader.dataset
        bs = getattr(base_loader, "batch_size", 32) or 32

        # just 1 batch worth of samples
        k = min(len(dataset), bs)
        idx = torch.randperm(len(dataset))[:k].tolist()

        return DataLoader(
            Subset(dataset, idx),
            batch_size=bs,
            shuffle=False,                 # we already sampled indices
            num_workers=0,                 # simple & robust
            pin_memory=getattr(base_loader, "pin_memory", True),
            drop_last=False,
        )

    # ---------------- Lightning hook: run once at fit start ----------------
    def on_fit_start(self):
        if self._gps_done:
            return

        assert hasattr(self, "model"), "GPS expects self.model (nn.Module)"
        assert hasattr(self, "criterion"), "GPS expects self.criterion (loss fn)"

        calib_loader = self._build_calibration_loader()

        # dummy optimizer to satisfy calculate_gradient's signature (no step is taken)
        dummy_opt = torch.optim.SGD(self.parameters(), lr=1e-3)

        # Some GPS versions read fields from args; keep it minimal & consistent
        args = SimpleNamespace(times_para=self.gps_percent)

        use_amp = str(getattr(self.trainer, "precision", "")).startswith("16")

        # 1) gradient probe (forward+backward over the tiny loader; no optimizer.step)
        #    Pass `self` (full LightningModule) so forward() returns logits,
        #    not raw backbone outputs (e.g. HuggingFace BaseModelOutputWithPooling).
        #    Gradients still flow to self.model.parameters() since it's a submodule.
        calculate_gradient(
            model=self,
            loader=calib_loader,
            optimizer=dummy_opt,
            loss_fn=self.criterion,
            amp_autocast=use_amp,
        )

        # 2) prune by percentile per cell (keep ~gps_percent%)
        prune_by_percentile_gradient_perCell(self.model, self.gps_percent)

        # 3) clear grads & rebuild optimizers so they only track unfrozen params
        self.model.zero_grad(set_to_none=True)
        self.trainer.strategy.setup_optimizers(self.trainer)

        self._gps_done = True
        if hasattr(self, "print"):
            self.print(f"[GPS] Pruned to top {self.gps_percent}% per cell; optimizers reinitialized.")


def prune_by_percentile_gradient_perCell(model, time_para=1):
    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "norm" in name or "pos_embed" in name or "cls_token" in name:
            new_mask = np.ones_like(param.data.cpu().numpy())
        elif 'head' in name or "bias" in name or "gamma" in name:
            new_mask = np.zeros_like(param.data.cpu().numpy())
        elif param.grad is None:
            # No gradient (e.g. pooler params unused in forward) → freeze
            new_mask = np.ones_like(param.data.cpu().numpy())
        else:
            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                tensor = param.grad.data.cpu().numpy()
                B,C,H,W = tensor.shape
                tensor = np.reshape(tensor,[B,-1])
            else:
                tensor = param.grad.data.cpu().numpy()

            new_mask=np.ones_like(tensor)
            for ind in range(time_para):
                max_index = abs(tensor).argsort(1)[:, -(ind + 1)]
                one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(np.bool)
                new_mask_temp = one_hot_temp.astype(np.float32)
                new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
                new_mask = new_mask.astype(np.float32)


            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                new_mask = np.reshape(new_mask, (B,C,H,W))

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )


        new_masks[name] = torch.from_numpy(new_mask).cuda()


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4) if total_withouthead else 0, "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4) if total_head else 0, "%)")
    total_all = total_head + total_withouthead
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_all, "(", np.round(((trainable_head+trainable_withouthead)/total_all)*100,4) if total_all else 0, "%)")

    print("#######################################################################")
    return new_masks


def calculate_gradient(model, loader, optimizer, loss_fn, amp_autocast=False):
    """
    Run forward+backward over `loader` to populate .grad on model params.
    - No optimizer.step()
    - No second-order graph
    - No distributed/logging/timm utils required
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    ctx = amp_autocast_ctx() if autocast_mode else nullcontext()
    device = next(model.parameters()).device

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with ctx:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        loss.backward()