from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from matplotlib.figure import Figure

from glovita.configs.logging import MlflowLoggerConfig
from glovita.logging.base import ExperimentLogger
from glovita.logging.utils import flatten_mapping, make_json_safe, mlflow_param_value, sanitize_artifact_name


class MlflowLogger(ExperimentLogger):
    """MLflow backend."""

    def __init__(
        self,
        config: MlflowLoggerConfig,
        *,
        default_experiment_name: str,
        default_tracking_uri: str | None = None,
        group: str,
        extra_tags: list[str] | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "Logger backend 'mlflow' was selected, but the 'mlflow' package is not installed. "
                "Install it with `pip install -e .[mlflow]`."
            ) from exc

        self._mlflow = mlflow
        tracking_uri = config.tracking_uri or default_tracking_uri
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = config.experiment_name or default_experiment_name
        mlflow.set_experiment(experiment_name)
        tags: dict[str, str] = {"group": config.group or group}
        if config.tags:
            tags["tags"] = ",".join(config.tags)
        if extra_tags:
            tags["extra_tags"] = ",".join(extra_tags)
        self._run = mlflow.start_run(run_name=config.run_name, tags=tags)

    def log_config(self, payload: Mapping[str, Any]) -> None:
        flat = flatten_mapping(make_json_safe(dict(payload)))
        params = {key: mlflow_param_value(value) for key, value in flat.items()}
        if params:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float | int], step: int | None = None) -> None:
        clean_metrics = {k: float(v) for k, v in metrics.items()}
        self._mlflow.log_metrics(clean_metrics, step=step)

    def log_figure(self, name: str, figure: Figure, step: int | None = None) -> None:
        safe_name = sanitize_artifact_name(name)
        artifact_file = f"figures/{safe_name}.png" if step is None else f"figures/step_{step:04d}_{safe_name}.png"
        self._mlflow.log_figure(figure, artifact_file)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def finish(self) -> None:
        self._mlflow.end_run()
