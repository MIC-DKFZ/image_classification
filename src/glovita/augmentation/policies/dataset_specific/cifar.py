from __future__ import annotations

import albumentations as A
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2

from glovita.augmentation.policies.adapters import AlbumentationsTransformAdapter
from glovita.augmentation.policies.metadata import TrainPolicySpec
from glovita.augmentation.policies.two_dim.randaugment import CIFAR10Policy, Cutout, RandAugment

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def build_baseline_transform(
    *,
    image_size: int = 32,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_baseline_cutout_transform(
    *,
    image_size: int = 32,
    cutout_size: int = 16,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_autoaugment_transform(
    *,
    image_size: int = 32,
    cutout_size: int = 16,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_train_transform(
    *,
    image_size: int = 32,
    cutout_size: int = 16,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_album_augment_transform(
    *,
    image_size: int = 32,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    padded_size = image_size + 4
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.InvertImg(p=0.2),
                A.PadIfNeeded(min_height=padded_size, min_width=padded_size, border_mode=4, p=0.2),
                A.RandomCrop(height=image_size, width=image_size, p=1.0),
                A.HorizontalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    p=0.2,
                    shift_limit_x=(-0.2, 0.2),
                    shift_limit_y=(-0.2, 0.2),
                    scale_limit=(0.0, 0.0),
                    rotate_limit=(0, 0),
                    interpolation=1,
                    border_mode=4,
                ),
                A.Equalize(p=0.2, mode="cv", by_channels=True),
                A.Solarize(p=0.2, threshold=(128, 128)),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


def build_test_transform(
    *,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_train_transform_dino(
    *,
    image_size: int = 224,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_test_transform_dino(
    *,
    image_size: int = 224,
    mean: tuple[float, ...] = MEAN,
    std: tuple[float, ...] = STD,
):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


SPATIAL_DIM = 2
TRAIN_POLICIES = {
    "baseline": TrainPolicySpec(build_baseline_transform),
    "baseline_cutout": TrainPolicySpec(build_baseline_cutout_transform, {"cutout_size": 16}),
    "autoaugment": TrainPolicySpec(build_autoaugment_transform, {"cutout_size": 16}),
    "randaugment": TrainPolicySpec(build_train_transform, {"cutout_size": 16}),
    "albumentations": TrainPolicySpec(build_album_augment_transform),
    "dino": TrainPolicySpec(build_train_transform_dino, {"image_size": 224}),
}
TEST_POLICIES = {
    "default": build_test_transform,
    "dino": build_test_transform_dino,
}
