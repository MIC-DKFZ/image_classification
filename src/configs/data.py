from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class BaseDataConfig(BaseModel):
    """Shared data-loading parameters for all datasets."""

    data_root_dir: Path
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=12, ge=0)
    # Subsample training data to this fraction (None = use full dataset)
    data_fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    # Cross-validation fold identifier (e.g. "0", "1"). None = use default split.
    fold: Optional[str] = None
    # If True, draw training batches with replacement (RandomSampler)
    random_batches: bool = False
    # Stratify the data-fraction subsample by class label
    stratified: bool = True


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
    batch_size: int = 128


class Cifar100Config(BaseDataConfig):
    dataset: Literal["cifar100"] = "cifar100"
    num_classes: int = 100
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 128


class ImagenetConfig(BaseDataConfig):
    dataset: Literal["imagenet"] = "imagenet"
    num_classes: int = 1000
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 8
    data_fraction: Optional[float] = 0.1


class PCamConfig(BaseDataConfig):
    dataset: Literal["pcam"] = "pcam"
    num_classes: int = 2
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class RxRx1Config(BaseDataConfig):
    dataset: Literal["rxrx1"] = "rxrx1"
    num_classes: int = 1139
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class NeuDetConfig(BaseDataConfig):
    dataset: Literal["neudet"] = "neudet"
    num_classes: int = 6
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class ZooScanNetConfig(BaseDataConfig):
    dataset: Literal["zooscannet"] = "zooscannet"
    num_classes: int = 20
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class AIDConfig(BaseDataConfig):
    dataset: Literal["aid"] = "aid"
    num_classes: int = 30
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class ChestXRay14Config(BaseDataConfig):
    dataset: Literal["chestxray14"] = "chestxray14"
    num_classes: int = 14
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multilabel"] = "multilabel"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class RESISC45Config(BaseDataConfig):
    dataset: Literal["resisc45"] = "resisc45"
    num_classes: int = 45
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class Flowers102Config(BaseDataConfig):
    dataset: Literal["flowers102"] = "flowers102"
    num_classes: int = 102
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class FGVCAircraftConfig(BaseDataConfig):
    dataset: Literal["fgvc_aircraft"] = "fgvc_aircraft"
    num_classes: int = 100
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


class DiabeticRetinaConfig(BaseDataConfig):
    dataset: Literal["diabetic_retina"] = "diabetic_retina"
    num_classes: int = 5
    task: Literal["Classification"] = "Classification"
    subtask: Literal["multiclass"] = "multiclass"
    batch_size: int = 32
    data_fraction: Optional[float] = 0.1


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
    ],
    Field(discriminator="dataset"),
]
