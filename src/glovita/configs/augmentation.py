from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, JsonValue


# Shared defaults
SharedTrainPolicy: TypeAlias = Literal[
    "default_2d_1",
    "default_2d_2",
    "default_2d_3",
    "default_2d_4",
    "default_2d_5",
    "default_2d_randaugment",
    "default_3d_1",
    "default_3d_2",
    "default_3d_3",
    "default_3d_4",
    "default_nnunet",
    "default_nnunet_DA5",
]
SharedTestPolicy: TypeAlias = Literal["shared_default_2d", "shared_default_3d"]

# Dataset-specific train policies currently exposed in the repo.
DatasetTrainPolicy: TypeAlias = Literal[
    "dataset_specific",
    "baseline",
    "baseline_cutout",
    "autoaugment",
    "randaugment",
    "albumentations",
    "dino",
    "timm_randaugment",
    "randaugment_448",
    "randaugment_224",
    "aid_large_crop",
    "native",
]
DatasetTestPolicy: TypeAlias = Literal[
    "default",
    "dino",
    "default_448",
    "default_224",
    "native",
]

TrainPolicyName: TypeAlias = SharedTrainPolicy | DatasetTrainPolicy
TestPolicyName: TypeAlias = SharedTestPolicy | DatasetTestPolicy


class AugmentationConfig(BaseModel):
    """User-facing augmentation settings.

    Train/test policy names are chosen from the policy sets implemented under
    `augmentation/policies`. Dataset configs provide sensible defaults so users
    can inspect the current defaults in `glovita.configs.data` and override only
    what they care about here.

    Runtime behavior:
    - `train_policy` and `test_policy` default per dataset in `DataConfig`.
    - encoder preprocessing defaults may populate `image_size`, `resize_size`,
      `mean`, `std`, and `patch_size` when these fields are left as `None`.
    - explicit values set here always win over encoder-derived defaults.
    """

    train_policy: TrainPolicyName | None = None
    test_policy: TestPolicyName | None = None
    image_size: int | None = Field(default=None, ge=1)
    resize_size: int | None = Field(default=None, ge=1)
    crop_size: int | None = Field(default=None, ge=1)
    cutout_size: int | None = Field(default=None, ge=1)
    patch_size: tuple[int, int, int] | None = None
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    # Escape hatches for policy-specific parameters that should override the
    # shared defaults separately for train and eval builders.
    train_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    test_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
