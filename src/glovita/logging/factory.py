from __future__ import annotations

from pathlib import Path

from glovita.configs.logging import LoggerConfig, MlflowLoggerConfig, NoLoggerConfig, WandbLoggerConfig
from glovita.logging.base import ExperimentLogger
from glovita.logging.mlflow_logger import MlflowLogger
from glovita.logging.null_logger import NullLogger
from glovita.logging.wandb_logger import WandbLogger


def build_logger(
    logger_config: LoggerConfig,
    *,
    log_dir: Path,
    default_experiment_name: str,
    default_tracking_uri: str | None = None,
    group: str,
    extra_tags: list[str] | None = None,
) -> ExperimentLogger:
    """Instantiate the selected experiment logger backend."""
    if isinstance(logger_config, WandbLoggerConfig):
        return WandbLogger(
            logger_config,
            log_dir=log_dir,
            default_project=default_experiment_name,
            group=group,
            extra_tags=extra_tags,
        )
    if isinstance(logger_config, MlflowLoggerConfig):
        return MlflowLogger(
            logger_config,
            default_experiment_name=default_experiment_name,
            default_tracking_uri=default_tracking_uri,
            group=group,
            extra_tags=extra_tags,
        )
    if isinstance(logger_config, NoLoggerConfig):
        return NullLogger()
    raise TypeError(f"Unsupported logger config: {type(logger_config).__name__}")
