from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from augmentation.policies.adapters import AlbumentationsTransformAdapter
from augmentation.policies.metadata import TrainPolicySpec
from augmentation.policies.two_dim import defaults as defaults_2d


def build_train_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    return defaults_2d.build_level3_train_transform(
        image_size=image_size,
        resize_size=resize_size,
        mean=mean,
        std=std,
    )


def build_test_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    return defaults_2d.build_test_transform(
        image_size=image_size,
        resize_size=resize_size,
        mean=mean,
        std=std,
    )


def build_train_transform_native(
    *,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.0,
                    p=0.5,
                ),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


def build_test_transform_native(
    *,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


SPATIAL_DIM = 2
TRAIN_POLICIES = {
    "dataset_specific": TrainPolicySpec(
        build_train_transform,
        {"image_size": 224, "resize_size": 256},
    ),
    "native": TrainPolicySpec(build_train_transform_native),
}
TEST_POLICIES = {
    "default": build_test_transform,
    "native": build_test_transform_native,
}
