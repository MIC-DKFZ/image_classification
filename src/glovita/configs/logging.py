from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class WandbLoggerConfig(BaseModel):
    """Weights & Biases logging configuration."""

    backend: Literal["wandb"] = "wandb"
    entity: str | None = Field(default=None, description="W&B entity/team name.")
    project: str | None = Field(default=None, description="W&B project name. Defaults to the dataset name when unset.")
    tags: list[str] | None = Field(default=None, description="Optional W&B tags.")
    group: str | None = Field(default=None, description="Optional run group. Defaults to an auto-generated experiment group.")
    name: str | None = Field(default=None, description="Optional explicit run name.")
    offline: bool = Field(default=False, description="Use W&B offline mode.")


class MlflowLoggerConfig(BaseModel):
    """MLflow logging configuration."""

    backend: Literal["mlflow"] = "mlflow"
    tracking_uri: str | None = Field(default=None, description="MLflow tracking URI. Defaults to the repo-wide experiment root when unset.")
    experiment_name: str | None = Field(default=None, description="MLflow experiment name. Defaults to the dataset name when unset.")
    run_name: str | None = Field(default=None, description="Optional explicit MLflow run name.")
    group: str | None = Field(default=None, description="Optional logical run group shared across folds.")
    tags: list[str] | None = Field(default=None, description="Optional MLflow tags.")


class NoLoggerConfig(BaseModel):
    """Disable experiment logging entirely."""

    backend: Literal["none"] = "none"


LoggerConfig = Annotated[
    Union[WandbLoggerConfig, MlflowLoggerConfig, NoLoggerConfig],
    Field(discriminator="backend"),
]
