import math
import warnings

import lightning as L
import torch
import torch.nn as nn
import wandb
from madgrad import MADGRAD
from timm.optim import RMSpropTF
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
    AveragePrecision,
)

from augmentation.mixup import mixup_criterion, mixup_data
from metrics.conf_mat import ConfusionMatrix
from regularization.sam import SAM


class BaseModel(L.LightningModule):
    def __init__(
        self,
        task,
        metric_computation_mode,
        result_plot,
        metrics,
        num_classes,
        name,
        lr,
        weight_decay,
        optimizer,
        nesterov,
        sam,
        adaptive_sam,
        scheduler,
        T_max,
        warmstart,
        epochs,
        mixup,
        mixup_alpha,
        label_smoothing,
        stochastic_depth,
        resnet_dropout,
        squeeze_excitation,
        apply_shakedrop,
        undecay_norm,
        zero_init_residual,
        input_dim,
        input_channels,
        pretrained,
        *args,
        **kwargs
    ):
        super(BaseModel, self).__init__()

        # Task
        self.task = task

        # Metrics
        self.metric_computation_mode = metric_computation_mode
        self.result_plot_setting = result_plot
        metrics_dict = {}

        self.subtask = kwargs["subtask"]

        if self.subtask == "multiclass":
            metric_task = "multiclass"
        elif self.subtask == "multilabel":
            metric_task = "multilabel"

        if self.task == "Classification":
            if "acc" in metrics:
                metrics_dict["Accuracy"] = Accuracy(
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )
            if "balanced_acc" in metrics:
                metrics_dict["Balanced_Accuracy"] = Accuracy(
                    task=metric_task,
                    num_classes=num_classes,
                    average="macro",
                    num_labels=num_classes,
                )
            if "f1" in metrics:
                metrics_dict["F1"] = F1Score(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "f1_per_class" in metrics:
                metrics_dict["F1_per_class"] = F1Score(
                    average=None,
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "pr" in metrics:
                metrics_dict["Precision"] = Precision(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
                metrics_dict["Recall"] = Recall(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "top5acc" in metrics:
                metrics_dict["Accuracy_top5"] = Accuracy(
                    task=metric_task,
                    num_classes=num_classes,
                    top_k=5,
                    num_labels=num_classes,
                )
            if "auroc" in metrics:
                metrics_dict["AUROC"] = AUROC(
                    average="macro",
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )
            if "ap" in metrics:
                metrics_dict["AP"] = AveragePrecision(
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )

        elif self.task == "Regression":
            if "mse" in metrics:
                metrics_dict["MSE"] = MeanSquaredError()
            if "mae" in metrics:
                metrics_dict["MAE"] = MeanAbsoluteError()

        if self.result_plot_setting in ["val", "all"]:
            if self.task == "Classification":
                self.val_conf_mat = ConfusionMatrix(num_classes=num_classes)
            elif self.task == "Regression":
                self.val_pred_list = []
                self.val_label_list = []
        if self.result_plot_setting == "all":
            if self.task == "Classification":
                self.train_conf_mat = ConfusionMatrix(num_classes=num_classes)
            elif self.task == "Regression":
                self.train_pred_list = []
                self.train_label_list = []

        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        # Training Args
        self.name = name
        # self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.nesterov = nesterov
        self.sam = sam
        self.adaptive_sam = adaptive_sam
        self.scheduler = scheduler
        self.T_max = T_max
        self.warmstart = warmstart
        self.warmstart2 = kwargs["warmstart2"]
        self.epochs = epochs
        self.pretrained = pretrained

        # Regularization techniques
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha  # 0.2
        self.label_smoothing = label_smoothing  # 0.1
        self.stochastic_depth = (
            stochastic_depth  # 0.1 (with higher resolution maybe 0.2)
        )
        self.resnet_dropout = resnet_dropout  # 0.5
        self.se = squeeze_excitation
        self.apply_shakedrop = apply_shakedrop
        self.undecay_norm = undecay_norm
        self.zero_init_residual = zero_init_residual

        # Finetuning method
        self.finetuning_method = kwargs["finetune_method"]

        # Data and Dataloading
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.num_classes = num_classes

        # switch to manual optimization for Sharpness Aware Minimization
        if self.sam:
            self.automatic_optimization = False

        # Loss
        if self.task == "Classification":
            if self.subtask == "multiclass":
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.label_smoothing
                )
            elif self.subtask == "multilabel":
                self.criterion = nn.BCEWithLogitsLoss()
        elif self.task == "Regression":
            self.criterion = nn.MSELoss()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):

        x, y = batch

        if self.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
            y_hat = self(inputs)

        else:
            y_hat = self(x)
            if self.num_classes == 1:
                y_hat = y_hat.view(-1)

        if self.sam:
            opt = self.optimizers()

            # first forward-backward pass
            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self.criterion(y_hat, y)
            self.manual_backward(loss)
            opt.first_step(zero_grad=True)

            # second forward-backward pass
            if self.mixup:
                self.manual_backward(
                    mixup_criterion(
                        self.criterion, self(inputs), targets_a, targets_b, lam
                    )
                )
            else:
                if self.num_classes == 1:
                    self.manual_backward(self.criterion(self(x).view(-1), y))
                else:
                    self.manual_backward(self.criterion(self(x), y))
            opt.second_step(zero_grad=True)

        else:
            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self.criterion(
                    y_hat, y.float() if self.subtask == "multilabel" else y
                )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if torch.isnan(y_hat).any():
            print("######################################### Model predicts NaNs!")

        # save metrics
        if self.metric_computation_mode == "stepwise":
            metrics_res = self.train_metrics(y_hat, y)
            if "train_F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["train_F1_per_class"]):
                    metrics_res["train_F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["train_F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        elif self.metric_computation_mode == "epochwise":
            self.train_metrics.update(y_hat, y)

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.update(y_hat, y)
        if hasattr(self, "train_pred_list"):
            self.train_pred_list.extend(y_hat)
            self.train_label_list.extend(y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.num_classes == 1:
            y_hat = y_hat.view(-1)

        val_loss = self.criterion(
            y_hat, y.float() if self.subtask == "multilabel" else y
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
        )

        # save metrics
        if self.metric_computation_mode == "stepwise":
            metrics_res = self.val_metrics(y_hat, y)
            if "val_F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["val_F1_per_class"]):
                    metrics_res["val_F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["val_F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )
        elif self.metric_computation_mode == "epochwise":
            self.val_metrics.update(y_hat, y)

        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.update(y_hat, y)
        if hasattr(self, "val_pred_list"):
            self.val_pred_list.extend(y_hat)
            self.val_label_list.extend(y)

    def predict_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        if self.num_classes == 1:
            y_hat = y_hat.view(-1)

        # self.predictions.append(y_hat)
        return y, y_hat

    def on_validation_epoch_end(self) -> None:
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.val_metrics.compute()
            if "val_F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["val_F1_per_class"]):
                    metrics_res["val_F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["val_F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )

            self.val_metrics.reset()

        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.save_state(self, "val")
            self.val_conf_mat.reset()
        if hasattr(self, "val_pred_list"):
            data = [[x, y] for (x, y) in zip(self.val_label_list, self.val_pred_list)]
            table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
            wandb.log(
                {
                    "Val Scatterplot": wandb.plot.scatter(
                        table, "Ground Truth", "Prediction", "Validation Scatterplot"
                    )
                }
            )
            # reset
            self.val_pred_list = []
            self.val_label_list = []

    def on_train_epoch_end(self) -> None:
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.train_metrics.compute()
            if "train_F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["train_F1_per_class"]):
                    metrics_res["train_F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["train_F1_per_class"]

            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )

            self.train_metrics.reset()

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.save_state(self, "train")
            self.train_conf_mat.reset()
        if hasattr(self, "train_pred_list"):
            data = [
                [x, y] for (x, y) in zip(self.train_label_list, self.train_pred_list)
            ]
            table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
            wandb.log(
                {
                    "Train Scatterplot": wandb.plot.scatter(
                        table, "Ground Truth", "Prediction", "Train Scatterplot"
                    )
                }
            )
            # reset
            self.train_pred_list = []
            self.train_label_list = []

    def on_train_start(self):
        from models.preact_resnet import PreActBlock, PreActBottleneck
        from models.pyramidnet import BasicBlock as BasicBlock_pyramid
        from models.pyramidnet import Bottleneck as Bottleneck_pyramid
        from models.resnet import BasicBlock, Bottleneck
        from models.wide_resnet import BasicBlock as Wide_BasicBlock
        from models.wide_resnet import Bottleneck as Wide_Bottleneck

        if not self.pretrained:
            print("Initializing weights")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    # nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            # TODO
            if self.zero_init_residual:
                if "PreAct" in self.name:
                    for m in self.modules():
                        if isinstance(m, PreActBottleneck):
                            nn.init.constant_(m.conv3.weight, 0)
                        elif isinstance(m, PreActBlock):
                            nn.init.constant_(m.conv2.weight, 0)

                elif "ResNet" in self.name or "WRN" in self.name:
                    for m in self.modules():
                        if isinstance(m, Bottleneck) or isinstance(m, Wide_Bottleneck):
                            nn.init.constant_(m.bn3.weight, 0)
                        elif isinstance(m, BasicBlock) or isinstance(
                            m, Wide_BasicBlock
                        ):
                            nn.init.constant_(m.bn2.weight, 0)

                elif "Pyramid" in self.name:
                    for m in self.modules():
                        if isinstance(m, Bottleneck_pyramid):
                            nn.init.constant_(m.bn4.weight, 0)
                        elif isinstance(m, BasicBlock_pyramid):
                            nn.init.constant_(m.bn3.weight, 0)

    def configure_optimizers(self):
        # leave bias and params of batch norm undecayed as in https://arxiv.org/pdf/1812.01187.pdf (Bag of tricks)
        if self.undecay_norm:
            model_params = []
            norm_params = []
            for name, p in self.named_parameters():
                if p.requires_grad:
                    if "norm" in name or "bias" in name or "bn" in name:
                        norm_params += [p]
                    else:
                        model_params += [p]
            params = [
                {"params": model_params},
                {"params": norm_params, "weight_decay": 0},
            ]
        else:
            params = self.parameters()

        if self.finetuning_method == "full_sawtooth":
            # Separate encoder and cls_head parameters
            encoder_params = []
            cls_head_params = []

            for name, param in self.named_parameters():
                if "encoder" in name:
                    encoder_params.append(param)
                elif "cls_head" in name:
                    cls_head_params.append(param)

        if not self.sam:
            if self.optimizer == "SGD":
                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.SGD(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "nesterov": self.nesterov,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "nesterov": self.nesterov,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.SGD(
                        params,
                        lr=self.lr,
                        momentum=0.9,
                        weight_decay=self.weight_decay,
                        nesterov=self.nesterov,
                    )
            elif self.optimizer == "Adam":
                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.Adam(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.Adam(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "AdamW":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.AdamW(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "Rmsprop":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = RMSpropTF(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = RMSpropTF(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "Madgrad":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = MADGRAD(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = MADGRAD(
                        params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
                    )

        else:
            # ASAM paper suggests 10x larger rho for adaptive SAM than in normal SAM
            rho = 0.5 if self.adaptive_sam else 0.05

            if self.optimizer == "SGD":
                base_optimizer = torch.optim.SGD
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                    nesterov=self.nesterov,
                    rho=rho,
                )
            elif self.optimizer == "Madgrad":
                base_optimizer = MADGRAD
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "Adam":
                base_optimizer = torch.optim.Adam
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "AdamW":
                base_optimizer = torch.optim.AdamW
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "Rmsprop":
                base_optimizer = RMSpropTF
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )

        if not self.scheduler:
            return [optimizer]
        else:
            if self.scheduler == "CosineAnneal" and self.warmstart == 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.T_max
                )
            elif self.scheduler == "CosineAnneal" and self.warmstart > 0:
                if self.finetuning_method == "full_sawtooth":
                    scheduler = CosineAnnealingLR_DoubleWarmstart(
                        optimizer,
                        T_max=self.T_max,
                        warmstart1=self.warmstart,
                        warmstart2=self.warmstart2,
                    )
                else:
                    scheduler = CosineAnnealingLR_Warmstart(
                        optimizer,
                        T_max=self.T_max,
                        warmstart=self.warmstart,
                    )
            elif self.scheduler == "Step":
                # decays every 1/4 epochs
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.epochs // 4, gamma=0.1
                )
            elif self.scheduler == "MultiStep":
                # decays lr with 0.1 after half of epochs and 3/4 of epochs
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, [self.epochs // 2, self.epochs * 3 // 4]
                )

            return [optimizer], [scheduler]


class CosineAnnealingLR_Warmstart(_LRScheduler):
    """
    Same as CosineAnnealingLR but includes a warmstart option that will gradually increase the LR
    for the amount of specified warmup epochs as described in https://arxiv.org/pdf/1706.02677.pdf
    """

    def __init__(
        self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, warmstart=0
    ):
        self.T_max = T_max - warmstart  # do not consider warmstart epochs for T_max
        self.eta_min = eta_min
        self.warmstart = warmstart
        self.T = 0

        super(CosineAnnealingLR_Warmstart, self).__init__(
            optimizer, last_epoch, verbose
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        # Warmstart
        if self.last_epoch < self.warmstart:
            addrates = [(lr / (self.warmstart + 1)) for lr in self.base_lrs]
            updated_lr = [
                addrates[i] * (self.last_epoch + 1)
                for i, group in enumerate(self.optimizer.param_groups)
            ]

            return updated_lr

        else:
            if self.T == 0:
                self.T += 1
                return self.base_lrs
            elif (self.T - 1 - self.T_max) % (2 * self.T_max) == 0:
                updated_lr = [
                    group["lr"]
                    + (base_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max))
                    / 2
                    for base_lr, group in zip(
                        self.base_lrs, self.optimizer.param_groups
                    )
                ]

                self.T += 1
                return updated_lr

            updated_lr = [
                (1 + math.cos(math.pi * self.T / self.T_max))
                / (1 + math.cos(math.pi * (self.T - 1) / self.T_max))
                * (group["lr"] - self.eta_min)
                + self.eta_min
                for group in self.optimizer.param_groups
            ]

            self.T += 1
            return updated_lr


class CosineAnnealingLR_DoubleWarmstart(_LRScheduler):
    """
    CosineAnnealingLR with two consecutive warmup phases.

    - Warmup 1: Increases LR from 0 to base LR, **only for `cls_head`**.
    - Warmup 2: Increases LR from 0 to base LR, **for both `cls_head` and `encoder`**.
    - Cosine Annealing: Decays LR **for both `cls_head` and `encoder`**.
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
        warmstart1=0,
        warmstart2=0,
    ):
        self.warmstart1 = warmstart1
        self.warmstart2 = warmstart2
        self.eta_min = eta_min
        self.T_max = T_max - (warmstart1 + warmstart2)  # Effective decay period
        self.T = 0  # Internal counter

        # Identify param groups: assume "cls_head" and "encoder" are named properly in optimizer param_groups
        self.cls_head_group = None
        self.encoder_group = None

        for param_group in optimizer.param_groups:
            if param_group.get("name") == "cls_head":
                self.cls_head_group = param_group
            elif param_group.get("name") == "encoder":
                self.encoder_group = param_group

        if self.cls_head_group is None:
            raise ValueError("Optimizer must have a parameter group named 'cls_head'.")
        if self.encoder_group is None:
            raise ValueError("Optimizer must have a parameter group named 'encoder'.")

        super(CosineAnnealingLR_DoubleWarmstart, self).__init__(
            optimizer, last_epoch, verbose
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        warmup_total = self.warmstart1 + self.warmstart2

        # First warmup phase (only `cls_head` is trained)
        if self.last_epoch < self.warmstart1:
            warmup_factor = (self.last_epoch + 1) / self.warmstart1
            updated_lr = []

            for group in self.optimizer.param_groups:
                if group is self.cls_head_group:
                    updated_lr.append(group["initial_lr"] * warmup_factor)
                else:  # Keep encoder frozen
                    updated_lr.append(0)

            return updated_lr

        # Second warmup phase (both `cls_head` and `encoder` are trained)
        elif self.last_epoch < warmup_total:
            warmup_factor = (self.last_epoch - self.warmstart1 + 1) / self.warmstart2
            updated_lr = [
                group["initial_lr"] * warmup_factor
                for group in self.optimizer.param_groups
            ]
            return updated_lr

        # Cosine annealing phase (both `cls_head` and `encoder`)
        else:
            epoch_cosine = self.last_epoch - warmup_total  # Shifted epoch count
            updated_lr = [
                self.eta_min
                + (group["initial_lr"] - self.eta_min)
                * 0.5
                * (1 + math.cos(math.pi * epoch_cosine / self.T_max))
                for group in self.optimizer.param_groups
            ]
            return updated_lr


class ModelConstructor(BaseModel):
    def __init__(self, model, **kwargs):
        super(ModelConstructor, self).__init__(**kwargs)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out
