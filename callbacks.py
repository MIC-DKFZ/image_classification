from lightning.pytorch.callbacks import TQDMProgressBar


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
