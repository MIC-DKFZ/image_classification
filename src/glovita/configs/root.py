from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue

from glovita.configs.data import DataConfig
from glovita.configs.dataloading import DataloadingConfig
from glovita.configs.model import ModelConfig
from glovita.configs.optimizer import OptimizerConfig
from glovita.configs.peft import PeftConfig
from glovita.configs.task import TaskConfig
from glovita.configs.training import TrainingConfig
from glovita.configs.wandb_cfg import WandbConfig


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
    dataloading: DataloadingConfig = Field(default_factory=DataloadingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    # Arbitrary logging-only metadata injected from CLI via --add_log.* flags.
    # This is saved and logged but never used to control runtime behavior.
    add_log: dict[str, JsonValue] = Field(default_factory=dict)

    # Root directory for experiment outputs (checkpoints, logs)
    exp_dir: Path = Path("./experiments")

    @property
    def default_wandb_project(self) -> str:
        """Default W&B project name derived from the selected dataset."""
        return getattr(self.data, "dataset", "default")

    def generate_wandb_group(self) -> str:
        """Generate a stable per-run default group when the user did not set one."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_name = type(self.model.encoder).__name__.replace("Config", "").lower()
        head_name = type(self.model.head).__name__.replace("Config", "").lower()
        peft_name = self.peft.method
        unique_id = str(uuid4())[:8]
        return f"{timestamp}_{encoder_name}_{head_name}_{peft_name}_{unique_id}"

    def resolve_wandb_kwargs(self) -> dict:
        """Return effective W&B init kwargs without mutating config state."""
        kwargs = self.wandb.model_dump(exclude_none=True)
        kwargs.setdefault("project", self.wandb.project or self.default_wandb_project)
        kwargs.setdefault("group", self.wandb.group or self.generate_wandb_group())
        return kwargs

    def get_run_log_dir(self, group: str) -> Path:
        """Per-run log directory: exp_dir / dataset / group."""
        dataset = getattr(self.data, "dataset", "unknown")
        return self.exp_dir / dataset / group
