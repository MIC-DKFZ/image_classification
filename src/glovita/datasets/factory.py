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
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

from glovita.augmentation.policies.registry import build_transforms, resolve_policy_names
from glovita.datasets.cifar import Cifar10Albumentation, Cifar100Albumentation
from glovita.datasets.generic_image_dataset import GenericImageDataset
from glovita.datasets.precomputed_features import PrecomputedFeaturesDataset
from glovita.datasets.utils import seed_worker
from glovita.configs.data import DataConfig
from glovita.configs.dataloading import DataloadingConfig


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


def _resolve_train_augmentation_kwargs(config: DataConfig) -> dict:
    return dict(config.augmentation.train_kwargs)


def _resolve_test_augmentation_kwargs(config: DataConfig) -> dict:
    return dict(config.augmentation.test_kwargs)


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
        "train_kwargs": _resolve_train_augmentation_kwargs(config),
        "test_kwargs": _resolve_test_augmentation_kwargs(config),
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
        train_overrides=_resolve_train_augmentation_kwargs(config),
        test_overrides=_resolve_test_augmentation_kwargs(config),
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )

    train_dataset_kwargs = dict(config.dataset_kwargs)
    train_dataset_kwargs.update(
        {
            "split": "train",
            "transform": train_transform,
            "split_file": split_file,
        }
    )
    val_dataset_kwargs = dict(config.dataset_kwargs)
    val_dataset_kwargs.update(
        {
            "split": "val",
            "transform": test_transform,
            "split_file": split_file,
        }
    )

    train_dataset = dataset_cls(
        config.data_root_dir,
        **train_dataset_kwargs,
    )
    val_dataset = dataset_cls(
        config.data_root_dir,
        **val_dataset_kwargs,
    )

    split_names = _load_split_names(Path(config.data_root_dir), split_file=split_file)
    if "test" in split_names:
        test_dataset_kwargs = dict(config.dataset_kwargs)
        test_dataset_kwargs.update(
            {
                "split": "test",
                "transform": test_transform,
                "split_file": split_file,
            }
        )
        test_dataset = dataset_cls(
            config.data_root_dir,
            **test_dataset_kwargs,
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
        train_overrides=_resolve_train_augmentation_kwargs(config),
        test_overrides=_resolve_test_augmentation_kwargs(config),
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )

    train_dataset_kwargs = dict(config.dataset_kwargs)
    train_dataset_kwargs.update(
        {
            "train": True,
            "transform": train_transform,
            "download": True,
        }
    )
    eval_dataset_kwargs = dict(config.dataset_kwargs)
    eval_dataset_kwargs.update(
        {
            "train": False,
            "transform": test_transform,
            "download": True,
        }
    )

    if "albumentations" in str(train_transform.__class__):
        train_dataset = Cifar10Albumentation(
            config.data_root_dir,
            **train_dataset_kwargs,
        )
    else:
        train_dataset = CIFAR10(
            config.data_root_dir,
            **train_dataset_kwargs,
        )

    if "albumentations" in str(test_transform.__class__):
        val_dataset = Cifar10Albumentation(
            config.data_root_dir,
            **eval_dataset_kwargs,
        )
    else:
        val_dataset = CIFAR10(
            config.data_root_dir,
            **eval_dataset_kwargs,
        )

    # CIFAR-10 has no official test split separate from val (the torchvision
    # "test" flag is the fixed 10k held-out set, which serves as both val and
    # test here).  Callers that need a true test set should use a dataset with
    # an explicit splits.json.
    return train_dataset, val_dataset, val_dataset


def _build_cifar100_datasets(
    config: DataConfig, encoder_preprocessing: dict | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(
        config.dataset,
        train_policy=config.augmentation.train_policy,
        test_policy=config.augmentation.test_policy,
        train_overrides=_resolve_train_augmentation_kwargs(config),
        test_overrides=_resolve_test_augmentation_kwargs(config),
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )

    train_dataset_kwargs = dict(config.dataset_kwargs)
    train_dataset_kwargs.update(
        {
            "train": True,
            "transform": train_transform,
            "download": True,
        }
    )
    eval_dataset_kwargs = dict(config.dataset_kwargs)
    eval_dataset_kwargs.update(
        {
            "train": False,
            "transform": test_transform,
            "download": True,
        }
    )

    if "albumentations" in str(train_transform.__class__):
        train_dataset = Cifar100Albumentation(
            config.data_root_dir,
            **train_dataset_kwargs,
        )
    else:
        train_dataset = CIFAR100(
            config.data_root_dir,
            **train_dataset_kwargs,
        )

    if "albumentations" in str(test_transform.__class__):
        val_dataset = Cifar100Albumentation(
            config.data_root_dir,
            **eval_dataset_kwargs,
        )
    else:
        val_dataset = CIFAR100(
            config.data_root_dir,
            **eval_dataset_kwargs,
        )

    # Same as CIFAR-10: val set doubles as the test split.
    return train_dataset, val_dataset, val_dataset


def _build_imagenet_datasets(
    config: DataConfig, encoder_preprocessing: dict | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(
        config.dataset,
        train_policy=config.augmentation.train_policy,
        test_policy=config.augmentation.test_policy,
        train_overrides=_resolve_train_augmentation_kwargs(config),
        test_overrides=_resolve_test_augmentation_kwargs(config),
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )
    train_dataset_kwargs = dict(config.dataset_kwargs)
    train_dataset_kwargs.update(
        {
            "split": "train",
            "transform": train_transform,
        }
    )
    eval_dataset_kwargs = dict(config.dataset_kwargs)
    eval_dataset_kwargs.update(
        {
            "split": "val",
            "transform": test_transform,
        }
    )
    train_dataset = ImageNet(
        config.data_root_dir,
        **train_dataset_kwargs,
    )
    val_dataset = ImageNet(
        config.data_root_dir,
        **eval_dataset_kwargs,
    )
    # ImageNet's official validation set is used as both val and test.
    return train_dataset, val_dataset, val_dataset


def _build_precomputed_feature_datasets(
    config: DataConfig, encoder_preprocessing: dict | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    _ = encoder_preprocessing
    train_dataset = PrecomputedFeaturesDataset(
        config.train_features_file,
        **config.dataset_kwargs,
    )
    val_dataset = PrecomputedFeaturesDataset(
        config.val_features_file,
        **config.dataset_kwargs,
    )
    if config.test_features_file is not None:
        test_dataset = PrecomputedFeaturesDataset(
            config.test_features_file,
            **config.dataset_kwargs,
        )
    else:
        test_dataset = val_dataset
    return train_dataset, val_dataset, test_dataset


def _build_generic_image_datasets(
    config: DataConfig, encoder_preprocessing: dict | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_transforms(
        config.dataset,
        train_policy=config.augmentation.train_policy,
        test_policy=config.augmentation.test_policy,
        train_overrides=_resolve_train_augmentation_kwargs(config),
        test_overrides=_resolve_test_augmentation_kwargs(config),
        **_resolve_augmentation_kwargs(config, encoder_preprocessing),
    )

    common_kwargs = dict(config.dataset_kwargs)
    common_kwargs.update(
        {
            "images_dir": config.images_dir,
            "split_source": config.split_source,
            "label_source": config.label_source,
            "split_file": config.split_file,
            "labels_file": config.labels_file,
            "path_column": config.path_column,
            "label_column": config.label_column,
            "train_split_name": config.train_split_name,
            "val_split_name": config.val_split_name,
            "test_split_name": config.test_split_name,
            "allowed_exts": config.allowed_exts,
            "class_names": config.class_names,
            "task": config.task,
            "strict": config.strict,
        }
    )

    train_dataset = GenericImageDataset(
        config.data_root_dir,
        split="train",
        transform=train_transform,
        **common_kwargs,
    )
    val_dataset = GenericImageDataset(
        config.data_root_dir,
        split="val",
        transform=test_transform,
        **common_kwargs,
    )

    if config.split_source == "subdirs":
        test_dir = Path(config.data_root_dir) / config.images_dir / config.test_split_name
        if test_dir.exists():
            test_dataset = GenericImageDataset(
                config.data_root_dir,
                split="test",
                transform=test_transform,
                **common_kwargs,
            )
        else:
            test_dataset = val_dataset
    else:
        split_names = _load_split_names(Path(config.data_root_dir), split_file=config.split_file)
        if config.test_split_name in split_names:
            test_dataset = GenericImageDataset(
                config.data_root_dir,
                split="test",
                transform=test_transform,
                **common_kwargs,
            )
        else:
            test_dataset = val_dataset

    return train_dataset, val_dataset, test_dataset


_DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(build_datasets=_build_cifar10_datasets),
    "cifar100": DatasetSpec(build_datasets=_build_cifar100_datasets),
    "imagenet": DatasetSpec(build_datasets=_build_imagenet_datasets),
    "generic_image_dataset": DatasetSpec(build_datasets=_build_generic_image_datasets),
    "precomputed_features": DatasetSpec(build_datasets=_build_precomputed_feature_datasets),
    "pcam": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.pcam",
            dataset_class="PCamData",
            encoder_preprocessing=enc,
        )
    ),
    "rxrx1": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.rxrx1",
            dataset_class="RxRx1Data",
            encoder_preprocessing=enc,
        )
    ),
    "neudet": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.neudet",
            dataset_class="NEUDETData",
            encoder_preprocessing=enc,
        )
    ),
    "zooscannet": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.zooscannet",
            dataset_class="ZooScanNetData",
            encoder_preprocessing=enc,
        )
    ),
    "aid": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.aid",
            dataset_class="AIDData",
            encoder_preprocessing=enc,
        )
    ),
    "chestxray14": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.chestxray14",
            dataset_class="ChestXray14Data",
            encoder_preprocessing=enc,
        )
    ),
    "resisc45": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.resisc45",
            dataset_class="RESISC45Data",
            encoder_preprocessing=enc,
        )
    ),
    "flowers102": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.flowers102",
            dataset_class="Flowers102Data",
            encoder_preprocessing=enc,
        )
    ),
    "fgvc_aircraft": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.fgvc_aircraft",
            dataset_class="FGVCAircraftData",
            encoder_preprocessing=enc,
        )
    ),
    "diabetic_retina": DatasetSpec(
        build_datasets=lambda cfg, enc=None: _build_generic_split_datasets(
            cfg,
            dataset_module="glovita.datasets.diabetic_retina",
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


def _unwrap_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _bag_collate_fn(batch):
    features, labels = zip(*batch, strict=True)
    labels = np.asarray([int(label) for label in labels], dtype=np.int64)

    if not features:
        raise ValueError("Cannot collate an empty batch of MIL bags.")

    if features[0].ndim != 2:
        raise ValueError(
            f"MIL bag collation expects each sample to be a 2D [N, D] tensor, got {tuple(features[0].shape)}."
        )

    max_instances = max(int(feature.shape[0]) for feature in features)
    feature_dim = int(features[0].shape[1])
    padded = features[0].new_zeros((len(features), max_instances, feature_dim))
    mask = torch.zeros((len(features), max_instances), dtype=torch.bool)
    for idx, bag in enumerate(features):
        bag_len = int(bag.shape[0])
        padded[idx, :bag_len] = bag
        mask[idx, :bag_len] = True

    return {"features": padded, "mask": mask}, torch.from_numpy(labels)


def _resolve_collate_fn(dataset):
    base_dataset = _unwrap_dataset(dataset)
    if getattr(base_dataset, "is_bag_dataset", False):
        return _bag_collate_fn
    return None


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
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(dataset), count, replace=False)

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
    collate_fn = _resolve_collate_fn(dataset)
    return DataLoader(
        dataset,
        batch_size=dataloading.batch_size,
        shuffle=dataloading.shuffle_train,
        drop_last=dataloading.drop_last_train,
        collate_fn=collate_fn,
        **_build_common_loader_kwargs(dataloading),
    )


def _build_eval_loader(dataloading: DataloadingConfig, dataset) -> DataLoader:
    collate_fn = _resolve_collate_fn(dataset)
    return DataLoader(
        dataset,
        batch_size=dataloading.effective_eval_batch_size,
        shuffle=dataloading.shuffle_eval,
        drop_last=dataloading.drop_last_eval,
        collate_fn=collate_fn,
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
