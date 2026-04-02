"""Dataset factory: registry-based dataset + dataloader construction.

This mirrors the model-side registry style:
  - one central registry controls dataset wiring
  - dataset implementations stay plain torch ``Dataset`` classes
  - no per-dataset DataModule class is required for the training runtime

To add a new dataset, you typically only need:
  1) a dataset class in ``datasets/<name>.py`` (split-aware for train/val[/test])
  2) ``build_train_transform`` / ``build_test_transform`` in
     ``augmentation/policies/dataset_specific/<name>.py`` or one of the shared
     2D/3D defaults
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

from augmentation.policies.registry import build_transforms, resolve_policy_names
from datasets.cifar import Cifar10Albumentation, Cifar100Albumentation
from datasets.precomputed_features import PrecomputedFeaturesDataset
from datasets.utils import seed_worker
from src.configs.data import DataConfig
from src.configs.dataloading import DataloadingConfig


DatasetBuilder = Callable[..., tuple[Dataset, Dataset, Dataset]]


@dataclass(frozen=True)
class DatasetSpec:
    build_datasets: DatasetBuilder


def _import_attr(module_path: str, attr: str):
    module = import_module(module_path)
    return getattr(module, attr)


def _resolve_augmentation_kwargs(config: DataConfig, encoder_preprocessing: dict | None) -> dict:
    """Merge encoder-derived preprocessing defaults with explicit data overrides."""
    kwargs = dict(encoder_preprocessing or {})
    augmentation = config.augmentation
    data_overrides = {
        "image_size": augmentation.image_size,
        "resize_size": augmentation.resize_size,
        "crop_size": augmentation.crop_size,
        "cutout_size": augmentation.cutout_size,
        "patch_size": augmentation.patch_size,
        "mean": augmentation.mean,
        "std": augmentation.std,
    }
    for key, value in data_overrides.items():
        if value is not None:
            kwargs[key] = value
    return kwargs


def resolve_augmentation_config(config: DataConfig, encoder_preprocessing: dict | None = None) -> dict:
    """Return the effective augmentation selection and override kwargs for logging."""
    if getattr(config, "dataset", None) == "precomputed_features":
        return {"train_policy": None, "test_policy": None, "kwargs": {}}
    train_policy = config.augmentation.train_policy
    test_policy = config.augmentation.test_policy
    resolved_train_policy, resolved_test_policy = resolve_policy_names(
        config.dataset,
        train_policy=train_policy,
        test_policy=test_policy,
    )
    return {
        "train_policy": resolved_train_policy,
        "test_policy": resolved_test_policy,
        "kwargs": _resolve_augmentation_kwargs(config, encoder_preprocessing),
    }


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
    encoder_preprocessing: dict | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    dataset_cls = _import_attr(dataset_module, dataset_class)
    train_transform, test_transform = build_transforms(
        config.dataset,
        train_policy=config.augmentation.train_policy,
        test_policy=config.augmentation.test_policy,
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )

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


def _build_cifar10_datasets(
    config: DataConfig, encoder_preprocessing: dict | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(
        config.dataset,
        train_policy=config.augmentation.train_policy,
        test_policy=config.augmentation.test_policy,
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )

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


def _build_cifar100_datasets(
    config: DataConfig, encoder_preprocessing: dict | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(
        config.dataset,
        train_policy=config.augmentation.train_policy,
        test_policy=config.augmentation.test_policy,
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )

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


def _build_imagenet_datasets(
    config: DataConfig, encoder_preprocessing: dict | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(
        config.dataset,
        train_policy=config.augmentation.train_policy,
        test_policy=config.augmentation.test_policy,
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )
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


def _build_precomputed_feature_datasets(
    config: DataConfig, encoder_preprocessing: dict | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    _ = encoder_preprocessing
    train_dataset = PrecomputedFeaturesDataset(config.train_features_file)
    val_dataset = PrecomputedFeaturesDataset(config.val_features_file)
    if config.test_features_file is not None:
        test_dataset = PrecomputedFeaturesDataset(config.test_features_file)
    else:
        test_dataset = val_dataset
    return train_dataset, val_dataset, test_dataset


_DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(build_datasets=_build_cifar10_datasets),
    "cifar100": DatasetSpec(build_datasets=_build_cifar100_datasets),
    "imagenet": DatasetSpec(build_datasets=_build_imagenet_datasets),
    "precomputed_features": DatasetSpec(build_datasets=_build_precomputed_feature_datasets),
    "pcam": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.pcam",
            dataset_class="PCamData",
            encoder_preprocessing=enc,
        )
    ),
    "rxrx1": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.rxrx1",
            dataset_class="RxRx1Data",
            encoder_preprocessing=enc,
        )
    ),
    "neudet": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.neudet",
            dataset_class="NEUDETData",
            encoder_preprocessing=enc,
        )
    ),
    "zooscannet": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.zooscannet",
            dataset_class="ZooScanNetData",
            encoder_preprocessing=enc,
        )
    ),
    "aid": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.aid",
            dataset_class="AIDData",
            encoder_preprocessing=enc,
        )
    ),
    "chestxray14": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.chestxray14",
            dataset_class="ChestXray14Data",
            encoder_preprocessing=enc,
        )
    ),
    "resisc45": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.resisc45",
            dataset_class="RESISC45Data",
            encoder_preprocessing=enc,
        )
    ),
    "flowers102": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.flowers102",
            dataset_class="Flowers102Data",
            encoder_preprocessing=enc,
        )
    ),
    "fgvc_aircraft": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.fgvc_aircraft",
            dataset_class="FGVCAircraftData",
            encoder_preprocessing=enc,
        )
    ),
    "diabetic_retina": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="datasets.diabetic_retina",
            dataset_class="EyePACSData",
            encoder_preprocessing=enc,
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


def _build_common_loader_kwargs(dataloading: DataloadingConfig) -> dict:
    """Translate `DataloadingConfig` into `torch.utils.data.DataLoader` kwargs."""
    kwargs = {
        "num_workers": dataloading.num_workers,
        "pin_memory": dataloading.pin_memory,
        "timeout": dataloading.timeout,
    }
    if dataloading.use_worker_init_fn:
        kwargs["worker_init_fn"] = seed_worker
    if dataloading.num_workers > 0:
        kwargs["persistent_workers"] = dataloading.effective_persistent_workers
        if dataloading.effective_prefetch_factor is not None:
            kwargs["prefetch_factor"] = dataloading.effective_prefetch_factor
    return kwargs


def _build_train_loader(dataloading: DataloadingConfig, dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=dataloading.batch_size,
        shuffle=dataloading.shuffle_train,
        drop_last=dataloading.drop_last_train,
        **_build_common_loader_kwargs(dataloading),
    )


def _build_eval_loader(dataloading: DataloadingConfig, dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=dataloading.effective_eval_batch_size,
        shuffle=dataloading.shuffle_eval,
        drop_last=dataloading.drop_last_eval,
        **_build_common_loader_kwargs(dataloading),
    )


def build_dataloaders(
    config: DataConfig,
    dataloading: DataloadingConfig,
    encoder_preprocessing: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)`` for the given config.

    Runtime flow:
    1. resolve dataset-specific train/test transforms
    2. build plain torch `Dataset` objects
    3. optionally subsample the training set
    4. wrap datasets with train/eval `DataLoader`s using `DataloadingConfig`
    """
    dataset_name = getattr(config, "dataset", None)
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"No dataset registry entry for dataset={dataset_name!r}")

    train_dataset, val_dataset, test_dataset = _DATASET_REGISTRY[dataset_name].build_datasets(
        config, encoder_preprocessing
    )
    train_dataset = _maybe_apply_fraction(train_dataset, config.data_fraction, config.stratified)

    train_loader = _build_train_loader(dataloading, train_dataset)
    val_loader = _build_eval_loader(dataloading, val_dataset)
    test_loader = _build_eval_loader(dataloading, test_dataset)
    return train_loader, val_loader, test_loader
