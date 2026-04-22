from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from matplotlib.figure import Figure

from glovita.logging.base import ExperimentLogger


class NullLogger(ExperimentLogger):
    """No-op logger backend."""

    def log_config(self, payload: Mapping[str, Any]) -> None:
        return

    def log_metrics(self, metrics: Mapping[str, float | int], step: int | None = None) -> None:
        return

    def log_figure(self, name: str, figure: Figure, step: int | None = None) -> None:
        return

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        return

    def finish(self) -> None:
        return
