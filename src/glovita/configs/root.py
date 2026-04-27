from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue, model_validator

from glovita.configs.data import DataConfig
from glovita.configs.dataloading import DataloadingConfig
from glovita.configs.logging import LoggerConfig, WandbLoggerConfig
from glovita.configs.model import ModelConfig
from glovita.configs.optimizer import OptimizerConfig
from glovita.configs.peft import FullFinetuningConfig, PeftConfig
from glovita.configs.task import TaskConfig
from glovita.configs.training import TrainingConfig


class RootConfig(BaseModel):
    """Top-level experiment configuration.

    CLI usage (tyro):
        glovita_train --data.dataset imagenet --data.data_root_dir /data/ILSVRC \
                      --model.encoder.encoder_type timm --model.encoder.type vit_base_patch16_224 \
                      --model.head.head_type classification \
                      --peft.method lora --peft.lora_rank 16 \
                      --training.epochs 20 --optimizer.lr 2e-5

    For subcommand-style selection of data / model / peft, tyro will automatically
    generate subcommands from the discriminated-union fields.
    """

    # Required: must be provided via CLI or config file
    data: DataConfig = Field(description="Dataset selection and dataset-specific defaults.")
    model: ModelConfig = Field(description="Encoder, head, and feature aggregation settings.")
    peft: PeftConfig = Field(default_factory=FullFinetuningConfig, description="Parameter-efficient fine-tuning method configuration. Defaults to full_finetuning.")

    # Optional blocks with sensible defaults
    task: TaskConfig = Field(default_factory=TaskConfig, description="Metric computation and task-level training options.")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training-loop and hardware settings.")
    dataloading: DataloadingConfig = Field(default_factory=DataloadingConfig, description="PyTorch DataLoader settings.")
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer and scheduler settings.")
    logger: LoggerConfig = Field(default_factory=WandbLoggerConfig, description="Experiment logger backend configuration.")
    # Arbitrary logging-only metadata injected from CLI via --add_log.* flags.
    # This is saved and logged but never used to control runtime behavior.
    add_log: dict[str, JsonValue] = Field(default_factory=dict, description="Logging-only metadata injected from the CLI via --add_log.* flags.")

    # Root directory for experiment outputs (checkpoints, logs)
    exp_dir: Path = Field(default=Path("./experiments"), description="Root directory for experiment outputs such as checkpoints and config snapshots.")

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_wandb_config(cls, data):
        if isinstance(data, dict) and "logger" not in data and "wandb" in data:
            data = dict(data)
            data["logger"] = data.pop("wandb")
        return data

    @property
    def default_logger_experiment_name(self) -> str:
        """Default experiment/project name derived from the selected dataset."""
        return getattr(self.data, "dataset", "default")

    @property
    def default_mlflow_tracking_uri(self) -> str:
        """Default local MLflow file store under the global experiment root."""
        mlflow_root = (self.exp_dir / "mlflow").resolve()
        mlflow_root.mkdir(parents=True, exist_ok=True)
        return mlflow_root.as_uri()

    def generate_run_group(self) -> str:
        """Generate a stable per-run default group when the user did not set one."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_name = type(self.model.encoder).__name__.replace("Config", "").lower()
        head_name = type(self.model.head).__name__.replace("Config", "").lower()
        peft_name = self.peft.method
        unique_id = str(uuid4())[:8]
        return f"{timestamp}_{encoder_name}_{head_name}_{peft_name}_{unique_id}"

    def get_run_log_dir(self, group: str) -> Path:
        """Per-run log directory: exp_dir / dataset / group."""
        dataset = getattr(self.data, "dataset", "unknown")
        return self.exp_dir / dataset / group
