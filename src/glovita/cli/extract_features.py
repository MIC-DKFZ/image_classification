from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import h5py
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm

from glovita.configs.cli import parse_cli
from glovita.configs.data import DataConfig
from glovita.configs.dataloading import DataloadingConfig
from glovita.configs.model import ModelConfig
from glovita.configs.peft import FullFinetuningConfig, PeftConfig
from glovita.configs.root import RootConfig
from glovita.datasets.factory import build_dataloaders
from glovita.models.factory import build_model
from glovita.models.feature_aggregator import aggregate_features
from glovita.models.peft.registry import apply_peft
from glovita.models.preprocessing import resolve_encoder_preprocessing_defaults


DEFAULT_OUTPUT_FILENAME_TEMPLATE = (
    "agg_{method}_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5"
)


class ExtractConfig(BaseModel):
    checkpoint_path: Path | None = None
    data: DataConfig | None = None
    model: ModelConfig | None = None
    peft: PeftConfig = Field(default_factory=FullFinetuningConfig)
    dataloading: DataloadingConfig = Field(default_factory=DataloadingConfig)
    method: Literal["cls_token", "avg", "sum", "mean_all", "joint"] = "joint"
    split: Optional[Literal["train", "val", "test"]] = None
    precision: Literal[16, 32] = 16
    compression: int = Field(default=4, ge=0, le=9)
    output_dir: Path = Path("./precomputed_features")
    output_filename: str | None = None
    use_eval_transform_for_train: bool = True


def _serialize_model_name(config: ModelConfig) -> str:
    encoder = config.encoder
    for attr in ("type", "variant", "encoder_type"):
        value = getattr(encoder, attr, None)
        if value is not None:
            return str(value).replace("/", "_").replace(".", "_")
    return type(encoder).__name__.replace("Config", "").lower()


def _make_extraction_data_config(config: ExtractConfig, data_config: DataConfig) -> DataConfig:
    if not config.use_eval_transform_for_train:
        return data_config
    augmentation = data_config.augmentation.model_copy(
        update={"train_policy": data_config.augmentation.test_policy}
    )
    return data_config.model_copy(update={"augmentation": augmentation})


def _feature_imgsize(batch_x: torch.Tensor) -> str:
    if batch_x.ndim >= 4:
        return str(int(batch_x.shape[-1]))
    return "na"


def _iter_requested_splits(config: ExtractConfig):
    if config.split is not None:
        return [config.split]
    return ["train", "val", "test"]


def _load_checkpoint_root_config(checkpoint_path: Path) -> RootConfig:
    run_dir = checkpoint_path.parent.parent
    config_file = run_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_file}. "
            "Expected a checkpoint created by this repository."
        )
    return RootConfig.model_validate_json(config_file.read_text())


def _resolve_runtime_config(config: ExtractConfig) -> tuple[DataConfig, ModelConfig, PeftConfig]:
    if config.checkpoint_path is not None:
        root_config = _load_checkpoint_root_config(config.checkpoint_path)
        data_config = config.data if config.data is not None else root_config.data
        model_config = config.model if config.model is not None else root_config.model
        peft_config = config.peft if config.peft.method != "full_finetuning" else root_config.peft
        return data_config, model_config, peft_config

    if config.data is None or config.model is None:
        raise ValueError(
            "Either provide --checkpoint-path, or provide both data and model config blocks."
        )
    return config.data, config.model, config.peft


def _load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model", state.get("state_dict"))
    if state_dict is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain a 'model' or 'state_dict' entry."
        )
    model.load_state_dict(state_dict)


def _resolve_output_filename(
    config: ExtractConfig,
    *,
    model_name: str,
    dataset_name: str,
    split: str,
    imgsize: str,
) -> str:
    template = config.output_filename or DEFAULT_OUTPUT_FILENAME_TEMPLATE
    checkpoint_stem = config.checkpoint_path.stem if config.checkpoint_path is not None else "none"
    return template.format(
        method=config.method,
        model=model_name,
        dataset=dataset_name,
        split=split,
        imgsize=imgsize,
        precision=config.precision,
        checkpoint=checkpoint_stem,
    )


@torch.no_grad()
def run_extraction(config: ExtractConfig) -> None:
    data_config, model_config, peft_config = _resolve_runtime_config(config)
    extraction_data_config = _make_extraction_data_config(config, data_config)
    encoder_preprocessing = resolve_encoder_preprocessing_defaults(model_config.encoder).as_kwargs()
    train_loader, val_loader, test_loader = build_dataloaders(
        extraction_data_config,
        config.dataloading,
        encoder_preprocessing=encoder_preprocessing,
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    model = build_model(model_config, output_dim=getattr(data_config, "num_classes", 1))
    model = apply_peft(model, peft_config)
    if config.checkpoint_path is not None:
        _load_checkpoint_weights(model, config.checkpoint_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = _serialize_model_name(model_config)
    dataset_name = getattr(data_config, "dataset", "unknown")

    for split in _iter_requested_splits(config):
        loader = loaders[split]
        loader_iter = iter(loader)
        try:
            first_batch = next(loader_iter)
        except StopIteration:
            print(f"Skipping empty split: {split}")
            continue

        x0, y0 = first_batch
        imgsize = _feature_imgsize(x0)
        first_features = aggregate_features(
            model.extract_features(x0.to(device)), config.method
        ).detach().cpu()
        feature_dim = int(first_features.shape[-1])
        dataset_len = len(loader.dataset)

        out_file = config.output_dir / _resolve_output_filename(
            config,
            model_name=model_name,
            dataset_name=dataset_name,
            split=split,
            imgsize=imgsize,
        )
        print(f"Saving {split} features to {out_file}")

        with h5py.File(out_file, "w") as f:
            dset_features = f.create_dataset(
                "features",
                shape=(dataset_len, feature_dim),
                dtype=f"float{config.precision}",
                chunks=(1, feature_dim),
                compression=config.compression,
            )
            dset_labels = f.create_dataset("labels", shape=(dataset_len,), dtype="int64")

            first_labels = y0.detach().cpu().numpy()
            first_features_np = first_features.numpy()
            first_batch_size = len(first_labels)
            dset_features[:first_batch_size] = first_features_np
            dset_labels[:first_batch_size] = first_labels
            index = first_batch_size

            for x, y in tqdm(loader_iter, desc=f"{split.upper()} Batches"):
                x = x.to(device)
                features = (
                    aggregate_features(model.extract_features(x), config.method)
                    .detach()
                    .cpu()
                    .numpy()
                )
                labels = y.detach().cpu().numpy()
                batch_size = len(labels)
                dset_features[index : index + batch_size] = features
                dset_labels[index : index + batch_size] = labels
                index += batch_size


def main(config: ExtractConfig | None = None) -> None:
    if config is None:
        config = parse_cli(ExtractConfig)
    run_extraction(config)


if __name__ == "__main__":
    main()

