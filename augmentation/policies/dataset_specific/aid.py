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
    resize_size: int = 640,
    crop_size: int = 600,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.SmallestMaxSize(max_size=resize_size),
                A.RandomCrop(height=crop_size, width=crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.0,
                    p=0.5,
                ),
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


def build_test_transform(
    *,
    image_size: int = 224,
    resize_size: int = 640,
    crop_size: int = 600,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.SmallestMaxSize(max_size=resize_size),
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


SPATIAL_DIM = 2
TRAIN_POLICIES = {
    "aid_large_crop": TrainPolicySpec(
        build_train_transform,
        {"image_size": 224, "resize_size": 640, "crop_size": 600},
    ),
}
TEST_POLICIES = {
    "default": build_test_transform,
}
