from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms

from glovita.augmentation.policies.adapters import AlbumentationsTransformAdapter
from glovita.augmentation.policies.metadata import TrainPolicySpec
from glovita.augmentation.policies.two_dim.randaugment import Cutout, RandAugment

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _eval_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.SmallestMaxSize(max_size=resize_size),
                A.CenterCrop(height=image_size, width=image_size),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


def build_level1_train_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    return _eval_transform(image_size=image_size, resize_size=resize_size, mean=mean, std=std)


def build_test_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    return _eval_transform(image_size=image_size, resize_size=resize_size, mean=mean, std=std)


def build_level2_train_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.SmallestMaxSize(max_size=resize_size),
                A.RandomCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=(-20, 20), border_mode=cv2.BORDER_REFLECT, p=0.3),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


def build_level3_train_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.SmallestMaxSize(max_size=resize_size),
                A.RandomCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=(-45, 45), border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                    p=0.5,
                ),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


def build_level4_train_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.03,
                    p=0.6,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                A.GaussNoise(std_range=(0.01, 0.04), p=0.15),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


def build_level5_train_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    _ = resize_size
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_REFLECT, p=0.6),
                A.Affine(scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), shear=(-8, 8), p=0.3),
                A.OneOf(
                    [
                        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04, p=1.0),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    ],
                    p=0.2,
                ),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 6),
                    hole_height_range=(0.05, 0.18),
                    hole_width_range=(0.05, 0.18),
                    fill=0,
                    p=0.2,
                ),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    )


def build_randaugment_train_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    cutout_size: int = 16,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


TRAIN_POLICIES: dict[str, TrainPolicySpec] = {
    "default_2d_1": TrainPolicySpec(build_level1_train_transform),
    "default_2d_2": TrainPolicySpec(build_level2_train_transform),
    "default_2d_3": TrainPolicySpec(build_level3_train_transform),
    "default_2d_4": TrainPolicySpec(build_level4_train_transform),
    "default_2d_5": TrainPolicySpec(build_level5_train_transform),
    "default_2d_randaugment": TrainPolicySpec(build_randaugment_train_transform),
}

TEST_POLICIES: dict[str, object] = {
    "shared_default_2d": build_test_transform,
}
