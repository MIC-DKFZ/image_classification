import math

from lightning.pytorch.callbacks import Callback, TQDMProgressBar


class NaNLossCallback(Callback):
    """Raises RuntimeError if train or val loss is NaN for `patience` consecutive iterations."""

    def __init__(self, patience: int = 3):
        self.patience = patience
        self._train_nan_count = 0
        self._val_nan_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None and (math.isnan(float(loss)) or math.isinf(float(loss))):
            self._train_nan_count += 1
            if self._train_nan_count >= self.patience:
                raise RuntimeError(
                    f"Training loss has been NaN/Inf for {self.patience} consecutive iterations. Aborting."
                )
        else:
            self._train_nan_count = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_nan_count = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None and (math.isnan(float(val_loss)) or math.isinf(float(val_loss))):
            self._val_nan_count += 1
            if self._val_nan_count >= self.patience:
                raise RuntimeError(
                    f"Validation loss has been NaN/Inf for {self.patience} consecutive iterations. Aborting."
                )
        else:
            self._val_nan_count = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_nan_count = 0


class ClusterTQDMProgressBar(TQDMProgressBar):
    """TQDMProgressBar with throttled output for cluster environments (LSF/SLURM).

    When cluster_mode=True: the first `initial_updates` visual update iterations
    print normally, and the remaining steps are throttled to at most `remaining_updates`
    total visual updates per epoch. This prevents flooding log emails.
    """

    def __init__(
        self,
        refresh_rate: int = 1,
        process_position: int = 0,
        leave: bool = False,
        cluster_mode: bool = True,
        initial_updates: int = 5,
        remaining_updates: int = 10,
    ):
        super().__init__(
            refresh_rate=refresh_rate,
            process_position=process_position,
            leave=leave,
        )
        self.cluster_mode = cluster_mode
        self.initial_updates = initial_updates
        self.remaining_updates = remaining_updates

    def _should_update(self, current: int, total: int) -> bool:
        if not self.is_enabled:
            return False
        if not self.cluster_mode or total is None or total == 0:
            return current % self.refresh_rate == 0 or current == total
        # First initial_updates visual iterations: always update
        if current <= self.initial_updates:
            return True
        if total <= self.initial_updates:
            return current == total
        # For remaining steps (initial_updates+1 .. total): throttle to remaining_updates updates
        remaining_steps = total - self.initial_updates
        for i in range(1, self.remaining_updates + 1):
            pos = self.initial_updates + round(i * remaining_steps / self.remaining_updates)
            if current == pos:
                return True
        return False
