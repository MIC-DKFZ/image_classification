from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, JsonValue, model_validator

from glovita.configs.augmentation import AugmentationConfig


class BaseDataConfig(BaseModel):
    """Shared data-loading parameters for all datasets."""

    data_root_dir: Path = Field(description="Dataset root directory.")
    # Subsample training data to this fraction (None = use full dataset).
    data_fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="Optional fraction of the training set to keep.")
    # Cross-validation fold identifier (e.g. "0", "1"). None = use default split.
    fold: Optional[str] = Field(default=None, description="Optional fold identifier. Dataset implementations decide how to interpret it.")
    # Stratify the data-fraction subsample by class label when subsampling.
    stratified: bool = Field(default=True, description="Stratify the data_fraction subsample by label when possible.")
    # User-facing augmentation defaults and overrides. Dataset-specific defaults
    # are defined on the concrete dataset config classes below.
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig, description="Augmentation defaults and explicit overrides for this dataset.")
    # Escape hatch for dataset-constructor-specific arguments that are not
    # worth promoting into the shared schema yet.
    dataset_kwargs: dict[str, JsonValue] = Field(default_factory=dict, description="Dataset-constructor-specific kwargs not covered by the shared schema.")


# ---------------------------------------------------------------------------
# Dataset-specific configs
# Each exposes the fixed metadata (num_classes, task, subtask) that the
# training loop needs, without requiring a separate TaskConfig.
# ---------------------------------------------------------------------------

class Cifar10Config(BaseDataConfig):
    dataset: Literal["cifar10"] = "cifar10"
    num_classes: int = 10
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="randaugment",
            test_policy="default",
        )
    )


class Cifar100Config(BaseDataConfig):
    dataset: Literal["cifar100"] = "cifar100"
    num_classes: int = 100
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="randaugment",
            test_policy="default",
        )
    )


class ImagenetConfig(BaseDataConfig):
    dataset: Literal["imagenet"] = "imagenet"
    num_classes: int = 1000
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="randaugment_448",
            test_policy="default_448",
        )
    )


class PCamConfig(BaseDataConfig):
    dataset: Literal["pcam"] = "pcam"
    num_classes: int = 2
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="dataset_specific",
            test_policy="shared_default_2d",
        )
    )


class RxRx1Config(BaseDataConfig):
    dataset: Literal["rxrx1"] = "rxrx1"
    num_classes: int = 1139
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="dataset_specific",
            test_policy="shared_default_2d",
        )
    )


class NeuDetConfig(BaseDataConfig):
    dataset: Literal["neudet"] = "neudet"
    num_classes: int = 6
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="dataset_specific",
            test_policy="shared_default_2d",
        )
    )


class ZooScanNetConfig(BaseDataConfig):
    dataset: Literal["zooscannet"] = "zooscannet"
    num_classes: int = 20
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="dataset_specific",
            test_policy="shared_default_2d",
        )
    )


class AIDConfig(BaseDataConfig):
    dataset: Literal["aid"] = "aid"
    num_classes: int = 30
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="aid_large_crop",
            test_policy="default",
        )
    )


class ChestXRay14Config(BaseDataConfig):
    dataset: Literal["chestxray14"] = "chestxray14"
    num_classes: int = 14
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multilabel"] = "multilabel"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="default_2d_2",
            test_policy="shared_default_2d",
        )
    )


class RESISC45Config(BaseDataConfig):
    dataset: Literal["resisc45"] = "resisc45"
    num_classes: int = 45
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="dataset_specific",
            test_policy="shared_default_2d",
        )
    )


class Flowers102Config(BaseDataConfig):
    dataset: Literal["flowers102"] = "flowers102"
    num_classes: int = 102
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="dataset_specific",
            test_policy="shared_default_2d",
        )
    )


class FGVCAircraftConfig(BaseDataConfig):
    dataset: Literal["fgvc_aircraft"] = "fgvc_aircraft"
    num_classes: int = 100
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="dataset_specific",
            test_policy="shared_default_2d",
        )
    )


class DiabeticRetinaConfig(BaseDataConfig):
    dataset: Literal["diabetic_retina"] = "diabetic_retina"
    num_classes: int = 5
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    data_fraction: Optional[float] = 0.1
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="dataset_specific",
            test_policy="shared_default_2d",
        )
    )


class GenericImageDatasetConfig(BaseDataConfig):
    """Reusable image-folder / split-file dataset for standard scalar-label image tasks."""

    dataset: Literal["generic_image_dataset"] = "generic_image_dataset"
    num_classes: int = Field(description="Number of classes for classification, or 1 for scalar regression.")
    task: Literal["Classification", "Regression"] = Field(default="Classification", description="Whether the generic dataset is used for classification or regression.")
    subtask: Literal["multiclass", "regression"] = Field(default="multiclass", description="Multiclass classification or scalar regression mode.")
    images_dir: str = Field(default="images", description="Directory under data_root_dir that contains the image files.")
    split_source: Literal["splits_json", "subdirs"] = Field(default="splits_json", description="Whether splits are defined by a JSON file or by train/val/test subdirectories.")
    label_source: Literal["folder", "json", "csv"] = Field(default="json", description="Where labels come from: class folders, a JSON mapping, or a CSV table.")
    split_file: str = Field(default="splits.json", description="Split-file name relative to data_root_dir when split_source='splits_json'.")
    labels_file: str = Field(default="labels.json", description="Label-file name relative to data_root_dir when using JSON or CSV labels.")
    path_column: str = Field(default="path", description="CSV column containing image-relative paths when label_source='csv'.")
    label_column: str = Field(default="label", description="CSV column containing labels when label_source='csv'.")
    train_split_name: str = Field(default="train", description="Split name used for the training set.")
    val_split_name: str = Field(default="val", description="Split name used for the validation set.")
    test_split_name: str = Field(default="test", description="Split name used for the test set.")
    allowed_exts: tuple[str, ...] = Field(default=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), description="File extensions considered valid image files.")
    class_names: list[str] | None = Field(default=None, description="Optional explicit class-name ordering for folder-based classification.")
    strict: bool = Field(default=True, description="Fail on missing files or malformed metadata instead of silently skipping bad entries.")
    augmentation: AugmentationConfig = Field(
        default_factory=lambda: AugmentationConfig(
            train_policy="default_2d_2",
            test_policy="shared_default_2d",
        )
    )

    @model_validator(mode="after")
    def _validate_task_settings(self):
        if self.task == "Regression":
            if self.subtask != "regression":
                raise ValueError("generic_image_dataset uses subtask='regression' for Regression tasks.")
            if self.num_classes != 1:
                raise ValueError("generic_image_dataset expects num_classes=1 for scalar regression targets.")
        elif self.subtask != "multiclass":
            raise ValueError("generic_image_dataset currently supports only subtask='multiclass' for Classification.")
        return self


class PrecomputedFeaturesConfig(BaseDataConfig):
    """Dataset config for training directly from extracted HDF5 feature files."""

    dataset: Literal["precomputed_features"] = "precomputed_features"
    num_classes: int = Field(description="Number of target classes represented by the feature files.")
    task: Literal["Classification"] = Field(default="Classification", description="Precomputed-feature training currently uses classification semantics.")
    subtask: Literal["multiclass"] = Field(default="multiclass", description="Subtask type for precomputed-feature training.")
    train_features_file: Path = Field(description="HDF5 file containing training features and labels.")
    val_features_file: Path = Field(description="HDF5 file containing validation features and labels.")
    test_features_file: Path | None = Field(default=None, description="Optional HDF5 file containing test features and labels.")


# ---------------------------------------------------------------------------
# Discriminated union over all datasets
# ---------------------------------------------------------------------------

DataConfig = Annotated[
    Union[
        Cifar10Config,
        Cifar100Config,
        ImagenetConfig,
        PCamConfig,
        RxRx1Config,
        NeuDetConfig,
        ZooScanNetConfig,
        AIDConfig,
        ChestXRay14Config,
        RESISC45Config,
        Flowers102Config,
        FGVCAircraftConfig,
        DiabeticRetinaConfig,
        GenericImageDatasetConfig,
        PrecomputedFeaturesConfig,
    ],
    Field(discriminator="dataset"),
]
