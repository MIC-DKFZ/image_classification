from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class WandbLoggerConfig(BaseModel):
    """Weights & Biases logging configuration."""

    backend: Literal["wandb"] = "wandb"
    entity: str | None = None
    project: str | None = None
    tags: list[str] | None = None
    group: str | None = None
    name: str | None = None
    offline: bool = False


class MlflowLoggerConfig(BaseModel):
    """MLflow logging configuration."""

    backend: Literal["mlflow"] = "mlflow"
    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] | None = None


class NoLoggerConfig(BaseModel):
    """Disable experiment logging entirely."""

    backend: Literal["none"] = "none"


LoggerConfig = Annotated[
    Union[WandbLoggerConfig, MlflowLoggerConfig, NoLoggerConfig],
    Field(discriminator="backend"),
]
