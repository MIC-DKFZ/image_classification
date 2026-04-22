from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from matplotlib.figure import Figure


class ExperimentLogger(ABC):
    """Minimal backend-neutral experiment logger interface."""

    @abstractmethod
    def log_config(self, payload: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float | int], step: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_figure(self, name: str, figure: Figure, step: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> None:
        raise NotImplementedError
