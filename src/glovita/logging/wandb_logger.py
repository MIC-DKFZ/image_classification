from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from matplotlib.figure import Figure

from glovita.configs.logging import WandbLoggerConfig
from glovita.logging.base import ExperimentLogger
from glovita.logging.utils import make_json_safe


class WandbLogger(ExperimentLogger):
    """Weights & Biases backend."""

    def __init__(
        self,
        config: WandbLoggerConfig,
        *,
        log_dir: Path,
        default_project: str,
        group: str,
        extra_tags: list[str] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Logger backend 'wandb' was selected, but the 'wandb' package is not installed. "
                "Install it with `pip install -e .[wandb]`."
            ) from exc

        self._wandb = wandb
        init_kwargs = config.model_dump(exclude_none=True)
        init_kwargs.pop("backend", None)
        init_kwargs.setdefault("project", config.project or default_project)
        init_kwargs.setdefault("group", config.group or group)
        offline = init_kwargs.pop("offline", False)
        if offline:
            init_kwargs["mode"] = "offline"
        init_kwargs["dir"] = str(log_dir)
        init_kwargs.setdefault("tags", [])
        init_kwargs["tags"] = list(init_kwargs["tags"] or [])
        if extra_tags:
            init_kwargs["tags"].extend(extra_tags)
        self._wandb.init(**init_kwargs)

    def log_config(self, payload: Mapping[str, Any]) -> None:
        self._wandb.config.update(make_json_safe(dict(payload)), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float | int], step: int | None = None) -> None:
        self._wandb.log(dict(metrics), step=step)

    def log_figure(self, name: str, figure: Figure, step: int | None = None) -> None:
        self._wandb.log({name: self._wandb.Image(figure)}, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._wandb.save(str(path), base_path=str(path.parent))

    def finish(self) -> None:
        self._wandb.finish()
