"""Main training loop built on HuggingFace Accelerate.

Design principles:
  - Explicit over magic: every step of the training loop is visible here.
  - All distributed / mixed-precision / gradient-accumulation concerns are
    handled by Accelerate; the rest of the code is plain PyTorch.
  - W&B is called directly (not through a logger abstraction).
  - NaN detection replaces the old NaNLossCallback.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from tqdm import tqdm
from torchmetrics import MetricCollection

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except Exception:
    Progress = None

from glovita.augmentation.mixup import mixup_criterion, mixup_data
from glovita.metrics.conf_mat import ConfusionMatrix
from glovita.models.peft.gps import maybe_apply_gps
from glovita.configs.data import DataConfig
from glovita.configs.task import TaskConfig
from glovita.configs.training import TrainingConfig


# ---------------------------------------------------------------------------
# Metric builder
# ---------------------------------------------------------------------------

def build_metrics(task_config: TaskConfig, data_config: DataConfig, prefix: str) -> MetricCollection:
    """Build a ``MetricCollection`` from the task + data configuration."""
    from torchmetrics import (
        AUROC, Accuracy, AveragePrecision, F1Score,
        MeanAbsoluteError, MeanSquaredError, Precision, Recall,
    )

    num_classes = data_config.num_classes
    task = data_config.task
    subtask = data_config.subtask
    metric_task = subtask  # "multiclass" | "multilabel"
    metrics_dict = {}

    if task == "Classification":
        if "acc" in task_config.metrics:
            metrics_dict["Accuracy"] = Accuracy(
                task=metric_task, num_classes=num_classes, num_labels=num_classes
            )
        if "balanced_acc" in task_config.metrics:
            metrics_dict["Balanced_Accuracy"] = Accuracy(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro",
            )
        if "f1" in task_config.metrics:
            metrics_dict["F1"] = F1Score(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro",
            )
        if "f1_per_class" in task_config.metrics:
            metrics_dict["F1_per_class"] = F1Score(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average=None,
            )
        if "pr" in task_config.metrics:
            metrics_dict["Precision"] = Precision(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro",
            )
            metrics_dict["Recall"] = Recall(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro",
            )
        if "top5acc" in task_config.metrics:
            metrics_dict["Accuracy_top5"] = Accuracy(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                top_k=5,
            )
        if "auroc" in task_config.metrics:
            metrics_dict["AUROC"] = AUROC(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro",
            )
        if "ap" in task_config.metrics:
            metrics_dict["AP"] = AveragePrecision(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
            )

    elif task == "Regression":
        if "mse" in task_config.metrics:
            metrics_dict["MSE"] = MeanSquaredError()
        if "mae" in task_config.metrics:
            metrics_dict["MAE"] = MeanAbsoluteError()

    return MetricCollection(metrics_dict, prefix=prefix)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Manages the full training lifecycle for a single fold.

    Usage::

        trainer = Trainer(training_cfg, task_cfg, data_cfg, optimizer_cfg)
        model = trainer.fit(model, optimizer, scheduler, train_loader, val_loader)
    """

    NAN_PATIENCE = 3  # consecutive NaN/Inf batches before aborting

    def __init__(
        self,
        training_config: TrainingConfig,
        task_config: TaskConfig,
        data_config: DataConfig,
        log_dir: Path,
    ):
        self.cfg = training_config
        self.task = task_config
        self.data = data_config
        self.log_dir = log_dir

        self.accelerator = Accelerator(
            mixed_precision=training_config.precision if training_config.precision != "no" else "no",
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        )

        # Criterion
        subtask = data_config.subtask
        task = data_config.task
        if task == "Classification":
            if subtask == "multiclass":
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=task_config.label_smoothing
                )
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        # Metrics (moved to device later in fit())
        self.train_metrics = build_metrics(task_config, data_config, prefix="train_")
        self.val_metrics = build_metrics(task_config, data_config, prefix="val_")

        # Optional confusion matrix / scatter tracking
        self.val_conf_mat: Optional[ConfusionMatrix] = None
        self.train_conf_mat: Optional[ConfusionMatrix] = None
        if task == "Classification":
            if task_config.result_plot in ("val", "all"):
                self.val_conf_mat = ConfusionMatrix(num_classes=data_config.num_classes)
            if task_config.result_plot == "all":
                self.train_conf_mat = ConfusionMatrix(num_classes=data_config.num_classes)

        # Regression scatter state
        self.val_preds: list = []
        self.val_labels: list = []
        self.train_preds: list = []
        self.train_labels: list = []

        # Best-checkpoint tracking: choose a logged validation metric key
        # explicitly when provided, otherwise fall back to the first logged one.
        self._best_metric_key = training_config.best_checkpoint_metric
        self._best_val_metric: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader,
        val_loader,
    ) -> nn.Module:
        """Train *model* and return the trained model."""

        # Compile before prepare so Accelerate wraps the compiled model
        if self.cfg.compile:
            model = torch.compile(model, mode="default")

        # Accelerate takes ownership of distribution / device placement
        if scheduler is not None:
            model, optimizer, train_loader, val_loader, scheduler = (
                self.accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
            )
        else:
            model, optimizer, train_loader, val_loader = (
                self.accelerator.prepare(model, optimizer, train_loader, val_loader)
            )

        maybe_apply_gps(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=self._loss_for_calibration,
            accelerator=self.accelerator,
        )

        # Move metrics to the right device
        device = self.accelerator.device
        self.train_metrics = self.train_metrics.to(device)
        self.val_metrics = self.val_metrics.to(device)
        if self.val_conf_mat is not None:
            self.val_conf_mat = self.val_conf_mat.to(device)
        if self.train_conf_mat is not None:
            self.train_conf_mat = self.train_conf_mat.to(device)

        # Optional sanity val
        if self.cfg.num_sanity_val_steps > 0:
            self._val_epoch(
                model,
                optimizer,
                val_loader,
                epoch=-1,
                sanity_steps=self.cfg.num_sanity_val_steps,
            )

        for epoch in range(self.cfg.epochs):
            self._train_epoch(model, optimizer, train_loader, epoch)
            self._val_epoch(model, optimizer, val_loader, epoch)
            if scheduler is not None:
                scheduler.step()
            if self.cfg.enable_checkpointing and self.accelerator.is_main_process:
                self._save_checkpoint(model, optimizer, epoch, is_best=False)

        return self.accelerator.unwrap_model(model)

    # ------------------------------------------------------------------
    # Internal loop methods
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader,
        epoch: int,
    ) -> None:
        model.train()
        total_loss = 0.0
        nan_count = 0
        mode = self.task.metric_computation_mode

        pbar = self._make_progress(
            loader,
            f"Epoch {epoch + 1}/{self.cfg.epochs} [train]",
            rich_style="cyan",
            split="train",
        )

        for x, y in pbar:
            with self.accelerator.accumulate(model):
                if self.task.mixup:
                    if not torch.is_tensor(x):
                        raise ValueError("Mixup currently only supports tensor inputs, not MIL bag dictionaries.")
                    x_mix, targets_a, targets_b, lam = mixup_data(x, y, alpha=self.task.mixup_alpha)
                    y_hat = model(x_mix)
                    loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
                else:
                    y_hat = model(x, target=y)
                    if self.data.num_classes == 1:
                        y_hat = y_hat.view(-1)
                    target = y.float() if self.data.subtask == "multilabel" else y
                    loss = self.criterion(y_hat, target)
                    aux_loss = self._get_model_aux_loss(model)
                    if aux_loss is not None:
                        loss = loss + aux_loss

                # NaN / Inf detection (replaces NaNLossCallback)
                if math.isnan(float(loss)) or math.isinf(float(loss)):
                    nan_count += 1
                    if nan_count >= self.NAN_PATIENCE:
                        raise RuntimeError(
                            f"Loss has been NaN/Inf for {self.NAN_PATIENCE} consecutive "
                            "batches — aborting training."
                        )
                else:
                    nan_count = 0

                self.accelerator.backward(loss)
                if self.cfg.gradient_clip_val is not None:
                    self.accelerator.clip_grad_norm_(
                        model.parameters(), self.cfg.gradient_clip_val
                    )
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.detach().float()

            gathered_y_hat, gathered_y = self._gather_for_metrics(y_hat, y)
            if mode == "stepwise":
                self._update_metrics(
                    self.train_metrics,
                    gathered_y_hat,
                    gathered_y,
                    self.train_conf_mat,
                )
                if self.task.result_plot == "all" and self.data.task == "Regression":
                    self.train_preds.extend(gathered_y_hat.detach().cpu().tolist())
                    self.train_labels.extend(gathered_y.detach().cpu().tolist())
            else:
                self.train_metrics.update(gathered_y_hat, gathered_y)

            pbar.set_postfix(**self._progress_postfix(loss, self.train_metrics, split="train"))

        avg_loss = (total_loss / len(loader)).item()

        if self.accelerator.is_main_process:
            log_dict = {"train_loss": avg_loss, "epoch": epoch}
            if mode == "epochwise":
                log_dict.update(self._compute_and_reset(self.train_metrics))
                if self.train_conf_mat is not None:
                    self.train_conf_mat.log_to_wandb("train", step=epoch)
                    self.train_conf_mat.reset()
                if self.data.task == "Regression" and self.train_preds:
                    self._log_scatter("Train", self.train_labels, self.train_preds)
                    self.train_preds, self.train_labels = [], []
            else:
                # stepwise: metrics already computed per-step; log epoch summary
                log_dict.update(self._compute_and_reset(self.train_metrics))
                if self.train_conf_mat is not None:
                    self.train_conf_mat.log_to_wandb("train", step=epoch)
                    self.train_conf_mat.reset()
            wandb.log(log_dict)

    def _val_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader,
        epoch: int,
        sanity_steps: int = 0,
    ) -> None:
        model.eval()
        total_loss = 0.0
        mode = self.task.metric_computation_mode

        with torch.no_grad():
            pbar = self._make_progress(
                loader,
                f"Epoch {epoch + 1}/{self.cfg.epochs} [val]",
                rich_style="magenta",
                split="val",
            )

            for step, (x, y) in enumerate(pbar):
                if sanity_steps > 0 and step >= sanity_steps:
                    break

                y_hat = model(x, target=y)
                if self.data.num_classes == 1:
                    y_hat = y_hat.view(-1)
                target = y.float() if self.data.subtask == "multilabel" else y
                loss = self.criterion(y_hat, target)
                aux_loss = self._get_model_aux_loss(model)
                if aux_loss is not None:
                    loss = loss + aux_loss
                total_loss += loss.detach().float()

                gathered_y_hat, gathered_y = self._gather_for_metrics(y_hat, y)
                if mode == "stepwise":
                    self._update_metrics(
                        self.val_metrics,
                        gathered_y_hat,
                        gathered_y,
                        self.val_conf_mat,
                    )
                    if self.task.result_plot in ("val", "all") and self.data.task == "Regression":
                        self.val_preds.extend(gathered_y_hat.detach().cpu().tolist())
                        self.val_labels.extend(gathered_y.detach().cpu().tolist())
                else:
                    self.val_metrics.update(gathered_y_hat, gathered_y)
                    if self.val_conf_mat is not None:
                        self.val_conf_mat.update(gathered_y_hat, gathered_y)

                pbar.set_postfix(**self._progress_postfix(loss, self.val_metrics, split="val"))

        if sanity_steps > 0:
            self.val_metrics.reset()
            return

        avg_loss = (total_loss / len(loader)).item()

        if self.accelerator.is_main_process:
            log_dict = {"val_loss": avg_loss, "epoch": epoch}
            metric_values = self._compute_and_reset(self.val_metrics)
            log_dict.update(metric_values)
            if self.val_conf_mat is not None:
                self.val_conf_mat.log_to_wandb("val", step=epoch)
                self.val_conf_mat.reset()
            if self.data.task == "Regression" and self.val_preds:
                self._log_scatter("Val", self.val_labels, self.val_preds)
                self.val_preds, self.val_labels = [], []
            wandb.log(log_dict)

            # Save best checkpoint when the selected validation metric improves.
            if self.cfg.enable_checkpointing and metric_values:
                metric_key = self._best_metric_key or next(iter(metric_values))
                if metric_key not in metric_values:
                    available = ", ".join(metric_values.keys())
                    raise KeyError(
                        f"training.best_checkpoint_metric={metric_key!r} was not found in "
                        f"the logged validation metrics. Available metrics: {available}"
                    )
                primary_val = metric_values[metric_key]
                higher_is_better = metric_key.casefold() not in {"mse", "mae"}
                if isinstance(primary_val, float):
                    is_best = (
                        self._best_val_metric is None
                        or (higher_is_better and primary_val > self._best_val_metric)
                        or (not higher_is_better and primary_val < self._best_val_metric)
                    )
                    if is_best:
                        self._best_val_metric = primary_val
                        self._save_checkpoint(model, optimizer, epoch, is_best=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_metrics(self, metrics, y_hat, y, conf_mat=None):
        metrics.update(y_hat, y)
        if conf_mat is not None:
            conf_mat.update(y_hat, y)

    def _get_model_aux_loss(self, model: nn.Module) -> torch.Tensor | None:
        aux_loss = getattr(model, "latest_aux_loss", None)
        if aux_loss is not None:
            return aux_loss
        wrapped = getattr(model, "module", None)
        if wrapped is not None:
            return getattr(wrapped, "latest_aux_loss", None)
        return None

    def _gather_for_metrics(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gathered_y_hat = self.accelerator.gather_for_metrics(y_hat.detach())
        gathered_y = self.accelerator.gather_for_metrics(y.detach())
        return gathered_y_hat, gathered_y

    def _compute_and_reset(self, metrics: MetricCollection) -> dict:
        result = {}
        raw = metrics.compute()
        for k, v in raw.items():
            if k.endswith("F1_per_class"):
                for i, val in enumerate(v):
                    result[k.replace("F1_per_class", f"F1_class_{i}")] = (
                        val.item() if not torch.isnan(val) else 0.0
                    )
            else:
                result[k] = v.item() if hasattr(v, "item") else float(v)
        metrics.reset()
        return result

    def _make_progress(
        self,
        loader,
        desc: str,
        rich_style: str | None = None,
        split: str | None = None,
    ):
        if not self.accelerator.is_local_main_process:
            return loader
        if self.cfg.cluster_progress_bar or not sys.stderr.isatty() or Progress is None:
            pbar = tqdm(loader, desc=desc, disable=False, dynamic_ncols=True)
            if self.cfg.cluster_progress_bar:
                return _ThrottledTqdm(pbar, total=len(loader))
            return pbar
        return _RichProgress(loader, desc=desc, total=len(loader), style=rich_style, split=split)

    def _progress_postfix(
        self,
        loss: torch.Tensor,
        metrics: MetricCollection,
        split: str,
        max_items: int = 3,
    ) -> dict[str, str]:
        postfix = {"split": split, "loss": f"{loss.item():.4f}"}
        try:
            raw = metrics.compute()
        except Exception:
            return postfix

        postfix["metric_1"] = ""
        postfix["metric_2"] = ""
        postfix["metric_3"] = ""
        count = 0
        for key, value in raw.items():
            if count >= max_items:
                break
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar = value.item()
            else:
                scalar = float(value)
            short_key = key.replace("train_", "").replace("val_", "")
            postfix[f"metric_{count + 1}"] = f"{short_key}={scalar:.4f}"
            count += 1
        return postfix

    def _log_scatter(self, split: str, labels: list, preds: list):
        data = [[x, y] for x, y in zip(labels, preds)]
        table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
        wandb.log(
            {f"{split} Scatterplot": wandb.plot.scatter(table, "Ground Truth", "Prediction")}
        )

    def _loss_for_calibration(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.data.num_classes == 1:
            outputs = outputs.view(-1)
        target = targets.float() if self.data.subtask == "multilabel" else targets
        return self.criterion(outputs, target)

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        is_best: bool = False,
    ):
        ckpt_dir = self.log_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "model": self.accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")
        # Always keep a 'last.pt' pointer
        torch.save(state, ckpt_dir / "last.pt")
        if is_best:
            torch.save(state, ckpt_dir / "best.pt")


# ---------------------------------------------------------------------------
# Cluster-friendly throttled tqdm wrapper
# ---------------------------------------------------------------------------

class _ThrottledTqdm:
    """Wraps a tqdm bar and throttles updates for SLURM/LSF log environments."""

    INITIAL = 5
    REMAINING = 10

    def __init__(self, pbar, total: int):
        self._pbar = pbar
        self._total = total
        self._step = 0

    def __iter__(self):
        for item in self._pbar:
            self._step += 1
            yield item

    def set_postfix(self, **kwargs):
        if self._should_update():
            self._pbar.set_postfix(**kwargs)

    def _should_update(self) -> bool:
        n = self._step
        total = self._total
        if n <= self.INITIAL:
            return True
        if total <= self.INITIAL:
            return n == total
        remaining_steps = total - self.INITIAL
        for i in range(1, self.REMAINING + 1):
            pos = self.INITIAL + round(i * remaining_steps / self.REMAINING)
            if n == pos:
                return True
        return False


class _RichProgress:
    """Rich-backed iterable with a tqdm-like ``set_postfix`` API."""

    def __init__(
        self,
        iterable,
        desc: str,
        total: int,
        style: str | None = None,
        split: str | None = None,
    ):
        self._iterable = iterable
        self._style = style
        self._split = split or ""
        rich_desc = f"[{style}]{desc}[/{style}]" if style else desc
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[loss]}", justify="left"),
            TextColumn("{task.fields[metric_1]}", justify="left"),
            TextColumn("{task.fields[metric_2]}", justify="left"),
            TextColumn("{task.fields[metric_3]}", justify="left"),
            transient=False,
        )
        self._task_id = self._progress.add_task(
            rich_desc,
            total=total,
            loss="",
            metric_1="",
            metric_2="",
            metric_3="",
        )

    def __iter__(self):
        with self._progress:
            for item in self._iterable:
                yield item
                self._progress.advance(self._task_id)

    def set_postfix(self, **kwargs):
        style = self._style

        def _fmt(text: str) -> str:
            if not text:
                return ""
            return f"[{style}]{text}[/{style}]" if style else text

        loss = kwargs.get("loss", "")
        self._progress.update(
            self._task_id,
            loss=_fmt(f"{self._split}_loss={loss}" if self._split else f"loss={loss}"),
            metric_1=_fmt(kwargs.get("metric_1", "")),
            metric_2=_fmt(kwargs.get("metric_2", "")),
            metric_3=_fmt(kwargs.get("metric_3", "")),
        )
