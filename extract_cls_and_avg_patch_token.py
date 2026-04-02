#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import h5py
import torch
import tyro
from pydantic import BaseModel, Field
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from datasets.factory import build_dataloaders
from models.factory import build_model
from models.feature_aggregator import AGGREGATION_METHODS, aggregate_features
from models.peft.registry import apply_peft
from models.preprocessing import resolve_encoder_preprocessing_defaults
from src.configs.data import DataConfig
from src.configs.dataloading import DataloadingConfig
from src.configs.model import ModelConfig
from src.configs.peft import PeftConfig


FNAME_FORMAT_FEATURES = "agg_{method}_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5"


class ExtractConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    peft: PeftConfig
    dataloading: DataloadingConfig = Field(default_factory=DataloadingConfig)
    method: Literal["cls_token", "avg", "sum", "mean_all", "joint"] = "joint"
    split: Optional[Literal["train", "val", "test"]] = None
    precision: Literal[16, 32] = 16
    compression: int = Field(default=4, ge=0, le=9)
    output_dir: Path = Path("./precomputed_features")
    use_eval_transform_for_train: bool = True


def _serialize_model_name(config: ModelConfig) -> str:
    encoder = config.encoder
    for attr in ("type", "variant", "encoder_type"):
        value = getattr(encoder, attr, None)
        if value is not None:
            return str(value).replace("/", "_").replace(".", "_")
    return type(encoder).__name__.replace("Config", "").lower()


def _make_extraction_data_config(config: ExtractConfig) -> DataConfig:
    if not config.use_eval_transform_for_train:
        return config.data
    augmentation = config.data.augmentation.model_copy(
        update={"train_policy": config.data.augmentation.test_policy}
    )
    return config.data.model_copy(update={"augmentation": augmentation})


def _feature_imgsize(batch_x: torch.Tensor) -> str:
    if batch_x.ndim >= 4:
        return str(int(batch_x.shape[-1]))
    return "na"


def _iter_requested_splits(config: ExtractConfig):
    if config.split is not None:
        return [config.split]
    return ["train", "val", "test"]


@torch.no_grad()
def main(config: ExtractConfig) -> None:
    data_config = _make_extraction_data_config(config)
    encoder_preprocessing = resolve_encoder_preprocessing_defaults(config.model.encoder).as_kwargs()
    train_loader, val_loader, test_loader = build_dataloaders(
        data_config,
        config.dataloading,
        encoder_preprocessing=encoder_preprocessing,
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    model = build_model(config.model, output_dim=getattr(config.data, "num_classes", 1))
    model = apply_peft(model, config.peft)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = _serialize_model_name(config.model)
    dataset_name = getattr(config.data, "dataset", "unknown")

    for split in _iter_requested_splits(config):
        loader = loaders[split]
        try:
            first_batch = next(iter(loader))
        except StopIteration:
            print(f"Skipping empty split: {split}")
            continue

        x0, _ = first_batch
        imgsize = _feature_imgsize(x0)
        sample_features = model.extract_features(x0.to(device))
        sample_agg = aggregate_features(sample_features, config.method).detach().cpu()
        feature_dim = int(sample_agg.shape[-1])
        dataset_len = len(loader.dataset)

        out_file = config.output_dir / FNAME_FORMAT_FEATURES.format(
            method=config.method,
            model=model_name,
            dataset=dataset_name,
            split=split,
            imgsize=imgsize,
            precision=config.precision,
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

            index = 0
            for x, y in tqdm(loader, desc=f"{split.upper()} Batches"):
                x = x.to(device)
                features = aggregate_features(model.extract_features(x), config.method).detach().cpu().numpy()
                labels = y.detach().cpu().numpy()
                batch_size = len(labels)
                dset_features[index : index + batch_size] = features
                dset_labels[index : index + batch_size] = labels
                index += batch_size


if __name__ == "__main__":
    main(tyro.cli(ExtractConfig))
