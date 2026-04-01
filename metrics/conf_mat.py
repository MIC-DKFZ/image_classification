from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.figure import Figure
from torchmetrics import Metric
from torchmetrics.utilities.data import _bincount


class ConfusionMatrix(Metric):
    full_state_update = False

    def __init__(self, num_classes: int, labels: list = None) -> None:
        """
        Create an empty confusion matrix
        Parameters
        ----------
        num_classes : int
            number of classes inside the Dataset
        labels : list of str, optional
            names of the labels in the dataset
        """

        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.arange(self.num_classes).astype(str)
        self.add_state(
            "mat",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def compute(self):
        return self.mat

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """
        updating the Confusion Matrix(self.mat)
        Parameters
        ----------
        pred : torch.Tensor
            prediction (softmax), with shape [batch_size, num_classes, height, width]
        gt : torch.Tensor
            gt mask, with shape [batch_size, height, width]
        """
        if pred.ndim > gt.ndim:
            pred = pred.argmax(dim=1)
        pred = pred.flatten()
        gt = gt.flatten()  # .detach()#.cpu()
        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            # Using the torchmetrics implementation of bincount, since the torch one does not support deterministic behaviour
            confmat = _bincount(inds, minlength=n**2).reshape(
                n, n
            )  # torch.bincount(inds, minlength=n**2).reshape(n, n)

        self.mat += confmat  # .to(self.mat)

    def _mat_to_figure(
        self,
        mat: np.ndarray,
        name: str = "Confusion matrix",
        norm_colorbar: bool = False,
    ) -> Figure:
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(mat, interpolation="nearest", cmap=plt.cm.viridis)
        plt.title(name)
        if norm_colorbar:
            plt.clim(0, 1)
        plt.colorbar()

        labels = self.labels if hasattr(self, "labels") else np.arange(self.num_classes)
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=0)
        plt.yticks(tick_marks, labels)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.close(figure)
        return figure

    def log_to_wandb(self, split: str, step: int | None = None) -> None:
        confmat = self.mat.detach().cpu().numpy()
        figure = self._mat_to_figure(confmat, "Confusion Matrix")

        row_sums = confmat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        confmat_norm = np.around(confmat.astype(float) / row_sums, decimals=2)
        figure_norm = self._mat_to_figure(
            confmat_norm,
            "Confusion Matrix (normalized)",
            norm_colorbar=True,
        )

        wandb.log(
            {
                f"{split}_ConfusionMatrix_absolute": wandb.Image(figure),
                f"{split}_ConfusionMatrix_normalized": wandb.Image(figure_norm),
            },
            step=step,
        )
