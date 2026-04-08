from __future__ import annotations

import torchvision.transforms as transforms
from timm.data import create_transform

from glovita.augmentation.policies.metadata import TrainPolicySpec
from glovita.augmentation.policies.two_dim.randaugment import ImageNetPolicy, RandAugment

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_baseline_transform(
    *,
    image_size: int = 224,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_autoaugment_transform(
    *,
    image_size: int = 224,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_timm_randaugment_transform(
    *,
    image_size: int = 448,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return create_transform(
        input_size=image_size,
        is_training=True,
        mean=mean,
        std=std,
        interpolation="bicubic",
        scale=(0.08, 1.0),
        ratio=(3 / 4, 4 / 3),
        auto_augment="rand-m9-mstd0.5",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )


def build_train_transform(
    *,
    image_size: int = 448,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_test_transform(
    *,
    image_size: int = 448,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_train_transform_224(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_test_transform_224(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


SPATIAL_DIM = 2
TRAIN_POLICIES = {
    "baseline": TrainPolicySpec(
        build_baseline_transform,
        {"image_size": 224, "resize_size": 256},
    ),
    "autoaugment": TrainPolicySpec(
        build_autoaugment_transform,
        {"image_size": 224, "resize_size": 256},
    ),
    "timm_randaugment": TrainPolicySpec(
        build_timm_randaugment_transform,
        {"image_size": 448},
    ),
    "randaugment_448": TrainPolicySpec(
        build_train_transform,
        {"image_size": 448},
    ),
    "randaugment_224": TrainPolicySpec(
        build_train_transform_224,
        {"image_size": 224, "resize_size": 256},
    ),
}
TEST_POLICIES = {
    "default_448": build_test_transform,
    "default_224": build_test_transform_224,
}
