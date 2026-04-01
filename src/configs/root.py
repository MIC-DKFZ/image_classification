from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from src.configs.data import DataConfig
from src.configs.model import ModelConfig
from src.configs.optimizer import OptimizerConfig
from src.configs.peft import PeftConfig
from src.configs.task import TaskConfig
from src.configs.training import TrainingConfig
from src.configs.wandb_cfg import WandbConfig


class RootConfig(BaseModel):
    """Top-level experiment configuration.

    CLI usage (tyro):
        python train.py --data.dataset imagenet --data.data_root_dir /data/ILSVRC \
                        --model.encoder.encoder_type timm --model.encoder.type vit_base_patch16_224 \
                        --model.head.head_type classification \
                        --peft.method lora --peft.lora_rank 16 \
                        --training.epochs 20 --optimizer.lr 2e-5

    For subcommand-style selection of data / model / peft, tyro will automatically
    generate subcommands from the discriminated-union fields.
    """

    # Required: must be provided via CLI or config file
    data: DataConfig
    model: ModelConfig
    peft: PeftConfig

    # Optional blocks with sensible defaults
    task: TaskConfig = Field(default_factory=TaskConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # Root directory for experiment outputs (checkpoints, logs)
    exp_dir: Path = Path("./experiments")

    @model_validator(mode="after")
    def auto_fill_wandb_group(self) -> "RootConfig":
        """Fill dynamic W&B defaults derived from the selected config."""
        if self.wandb.project is None:
            self.wandb.project = getattr(self.data, "dataset", "default")

        if self.wandb.group is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            encoder_name = type(self.model.encoder).__name__.replace("Config", "").lower()
            head_name = type(self.model.head).__name__.replace("Config", "").lower()
            peft_name = self.peft.method
            unique_id = str(uuid4())[:8]
            self.wandb.group = f"{timestamp}_{encoder_name}_{head_name}_{peft_name}_{unique_id}"
        return self

    @property
    def run_log_dir(self) -> Path:
        """Per-run log directory: exp_dir / dataset / group."""
        dataset = getattr(self.data, "dataset", "unknown")
        group = self.wandb.group or "default"
        return self.exp_dir / dataset / group
