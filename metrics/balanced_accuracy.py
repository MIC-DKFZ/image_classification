import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import stat_scores


class BalancedAccuracy(Metric):
    def __init__(
        self,
        num_classes: int,
        task: str = "multiclass",
        threshold: float = 0.5,
        dist_sync_on_step=False,
    ):
        assert task in {
            "multiclass",
            "multilabel",
        }, "Only 'multiclass' and 'multilabel' tasks are supported."
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.task = task
        self.threshold = threshold

        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        target = target.to(torch.long)
        if self.task == "multilabel":

            # Auto-detect logits vs probs
            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = torch.sigmoid(preds)

            preds = (preds >= self.threshold).long()

            stats = stat_scores(
                preds=preds,
                target=target,
                task="multilabel",
                num_labels=self.num_classes,
                average=None,
            )
        elif self.task == "multiclass":
            if preds.ndim == 2 and preds.size(1) == self.num_classes:
                preds = torch.argmax(preds, dim=1)

            stats = stat_scores(
                preds=preds,
                target=target,
                task="multiclass",
                num_classes=self.num_classes,
                average=None,
            )

        if stats.ndim == 1:
            stats = stats.unsqueeze(0)  # make it 2D to unbind along dim=1

        tp, fp, tn, fn, _ = stats.unbind(dim=1)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        recall = self.tp / (self.tp + self.fn + 1e-8)
        specificity = self.tn / (self.tn + self.fp + 1e-8)
        balanced_acc = (recall + specificity) / 2
        return balanced_acc.mean()
