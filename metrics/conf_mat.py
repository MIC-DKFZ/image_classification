import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
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
        pass

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
        # if softmax input
        pred = pred.argmax(1).flatten()  # .detach()#.cpu()
        # if argmax input
        # pred = pred.flatten()  # .detach()#.cpu()
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

    def save_state(self, trainer: pl.Trainer, split: str) -> None:
        """
        save the raw and normalized confusion matrix (self.mat) as image/figure
        Parameters
        ----------
        trainer : pl.Trainer
            The trainer itself to access the logger and parameters like current epoch etc.
        split : str
            Prefix that will be added to ConfusionMatrix, train / val / test
        """

        def mat_to_figure(mat: np.ndarray, name: str = "Confusion matrix", norm_colorbar=False) -> Figure:
            """
            Parameters
            ----------
            mat: np.ndarray
                Confusion Matrix as np array of shape n x n, with n = number of classes
            name: str, optional
                title of the image
            norm_colorbar: bool, optional
                if True, colorbar range will be set to 0-1
            Returns
            -------
            Figure:
                Visualization of the Confusion matrix
            """

            figure = plt.figure(figsize=(8, 8))
            plt.imshow(mat, interpolation="nearest", cmap=plt.cm.viridis)
            plt.title(name)
            if norm_colorbar:
                plt.clim(0, 1)
            plt.colorbar()
            if hasattr(self, "class_names"):
                labels = self.class_names
            else:
                labels = np.arange(self.num_classes)

            tick_marks = np.arange(len(labels))

            plt.xticks(tick_marks, labels, rotation=0)  # , rotation=-90   , 45)
            plt.yticks(tick_marks, labels)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.close(figure)

            return figure

        # Confusion Matrix
        confmat = self.mat.detach().cpu().numpy()
        figure = mat_to_figure(confmat, "Confusion Matrix")

        # Normalized Confusion Matrix
        confmat_norm = np.around(confmat.astype("float") / confmat.sum(axis=1)[:, np.newaxis], decimals=2)
        figure_norm = mat_to_figure(confmat_norm, "Confusion Matrix (normalized)", norm_colorbar=True)

        for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
            if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                logger.experiment.add_figure(
                    "{}_ConfusionMatrix_normalized/ConfusionMatrix".format(split),
                    figure_norm,
                    trainer.current_epoch,
                )
                logger.experiment.add_figure(
                    "{}_ConfusionMatrix_absolute/ConfusionMatrix".format(split),
                    figure,
                    trainer.current_epoch,
                )
            elif isinstance(logger, pl.loggers.mlflow.MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=figure_norm,
                    artifact_file="{}_ConfusionMatrix_normalized.png".format(split),
                )
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=figure,
                    artifact_file="{}_ConfusionMatrix_absolute.png".format(split),
                )
            elif isinstance(logger, pl.loggers.wandb.WandbLogger):
                logger.log_image(
                    key="{}_ConfusionMatrix_normalized".format(split),
                    images=[figure_norm],
                    step=trainer.current_epoch,
                )
                logger.log_image(
                    key="{}_ConfusionMatrix_absolute".format(split),
                    images=[figure],
                    step=trainer.current_epoch,
                )
