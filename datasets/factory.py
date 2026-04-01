"""Dataset factory: registry-based dataset + dataloader construction.

This mirrors the model-side registry style:
  - one central registry controls dataset wiring
  - dataset implementations stay plain torch ``Dataset`` classes
  - no per-dataset DataModule class is required for the training runtime

To add a new dataset, you typically only need:
  1) a dataset class in ``datasets/<name>.py`` (split-aware for train/val[/test])
  2) ``build_train_transform`` / ``build_test_transform`` in ``augmentation/policies/<name>.py``
  3) one entry in ``_DATASET_REGISTRY`` below
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

from augmentation.policies.registry import build_transforms
from datasets.cifar import Cifar10Albumentation, Cifar100Albumentation
from datasets.utils import seed_worker
from src.configs.data import DataConfig


DatasetBuilder = Callable[[DataConfig], tuple[Dataset, Dataset, Dataset]]


@dataclass(frozen=True)
class DatasetSpec:
    build_datasets: DatasetBuilder


def _import_attr(module_path: str, attr: str):
    module = import_module(module_path)
    return getattr(module, attr)


def _load_split_names(root_dir: Path, split_file: str = "splits.json") -> set[str]:
    split_path = root_dir / split_file
    if not split_path.exists():
        return set()
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return set()
    return set(payload.keys())


def _build_generic_split_datasets(
    config: DataConfig,
    *,
    dataset_module: str,
    dataset_class: str,
    split_file: str = "splits.json",
) -> tuple[Dataset, Dataset, Dataset]:
    dataset_cls = _import_attr(dataset_module, dataset_class)
    train_transform, test_transform = build_transforms(config.dataset)

    train_dataset = dataset_cls(
        config.data_root_dir,
        split="train",
        transform=train_transform,
        split_file=split_file,
    )
    val_dataset = dataset_cls(
        config.data_root_dir,
        split="val",
        transform=test_transform,
        split_file=split_file,
    )

    split_names = _load_split_names(Path(config.data_root_dir), split_file=split_file)
    if "test" in split_names:
        test_dataset = dataset_cls(
            config.data_root_dir,
            split="test",
            transform=test_transform,
            split_file=split_file,
        )
    else:
        test_dataset = val_dataset

    return train_dataset, val_dataset, test_dataset


def _build_cifar10_datasets(config: DataConfig) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(config.dataset)

    if "albumentations" in str(train_transform.__class__):
        train_dataset = Cifar10Albumentation(
            config.data_root_dir,
            train=True,
            transform=train_transform,
            download=True,
        )
    else:
        train_dataset = CIFAR10(
            config.data_root_dir,
            train=True,
            transform=train_transform,
            download=True,
        )

    if "albumentations" in str(test_transform.__class__):
        val_dataset = Cifar10Albumentation(
            config.data_root_dir,
            train=False,
            transform=test_transform,
            download=True,
        )
    else:
        val_dataset = CIFAR10(
            config.data_root_dir,
            train=False,
            transform=test_transform,
            download=True,
        )

    return train_dataset, val_dataset, val_dataset


def _build_cifar100_datasets(config: DataConfig) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(config.dataset)

    if "albumentations" in str(train_transform.__class__):
        train_dataset = Cifar100Albumentation(
            config.data_root_dir,
            train=True,
            transform=train_transform,
            download=True,
        )
    else:
        train_dataset = CIFAR100(
            config.data_root_dir,
            train=True,
            transform=train_transform,
            download=True,
        )

    if "albumentations" in str(test_transform.__class__):
        val_dataset = Cifar100Albumentation(
            config.data_root_dir,
            train=False,
            transform=test_transform,
            download=True,
        )
    else:
        val_dataset = CIFAR100(
            config.data_root_dir,
            train=False,
            transform=test_transform,
            download=True,
        )

    return train_dataset, val_dataset, val_dataset


def _build_imagenet_datasets(config: DataConfig) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(config.dataset)
    train_dataset = ImageNet(
        config.data_root_dir,
        split="train",
        transform=train_transform,
    )
    val_dataset = ImageNet(
        config.data_root_dir,
        split="val",
        transform=test_transform,
    )
    return train_dataset, val_dataset, val_dataset


_DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(build_datasets=_build_cifar10_datasets),
    "cifar100": DatasetSpec(build_datasets=_build_cifar100_datasets),
    "imagenet": DatasetSpec(build_datasets=_build_imagenet_datasets),
    "pcam": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.pcam",
            dataset_class="PCamData",
        )
    ),
    "rxrx1": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.rxrx1",
            dataset_class="RxRx1Data",
        )
    ),
    "neudet": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.neudet",
            dataset_class="NEUDETData",
        )
    ),
    "zooscannet": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.zooscannet",
            dataset_class="ZooScanNetData",
        )
    ),
    "aid": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.aid",
            dataset_class="AIDData",
        )
    ),
    "chestxray14": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.chestxray14",
            dataset_class="ChestXray14Data",
        )
    ),
    "resisc45": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.resisc45",
            dataset_class="RESISC45Data",
        )
    ),
    "flowers102": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.flowers102",
            dataset_class="Flowers102Data",
        )
    ),
    "fgvc_aircraft": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.fgvc_aircraft",
            dataset_class="FGVCAircraftData",
        )
    ),
    "diabetic_retina": DatasetSpec(
        build_datasets=lambda cfg: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.diabetic_retina",
            dataset_class="EyePACSData",
        )
    ),
}


def _extract_targets(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent_targets = _extract_targets(dataset.dataset)
        return np.asarray(parent_targets)[np.asarray(dataset.indices)]
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    raise AttributeError(
        f"{dataset.__class__.__name__} does not expose .targets or .labels for stratified sampling."
    )


def _maybe_apply_fraction(dataset, fraction: float | None, stratified: bool):
    if fraction is None or fraction >= 1.0:
        return dataset

    if stratified:
        targets = _extract_targets(dataset)
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=42)
        train_idx, _ = next(splitter.split(np.zeros(len(targets)), targets))
        indices = train_idx
    else:
        count = int(len(dataset) * fraction)
        indices = np.random.choice(len(dataset), count, replace=False)

    return Subset(dataset, indices)


def _build_train_loader(config: DataConfig, dataset) -> DataLoader:
    persistent = config.num_workers > 0
    if config.random_batches:
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=len(dataset),
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=persistent,
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=persistent,
    )


def _build_eval_loader(config: DataConfig, dataset) -> DataLoader:
    persistent = config.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=persistent,
    )


def build_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)`` for the given config."""
    dataset_name = getattr(config, "dataset", None)
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"No dataset registry entry for dataset={dataset_name!r}")

    train_dataset, val_dataset, test_dataset = _DATASET_REGISTRY[dataset_name].build_datasets(config)
    train_dataset = _maybe_apply_fraction(train_dataset, config.data_fraction, config.stratified)

    train_loader = _build_train_loader(config, train_dataset)
    val_loader = _build_eval_loader(config, val_dataset)
    test_loader = _build_eval_loader(config, test_dataset)
    return train_loader, val_loader, test_loader
