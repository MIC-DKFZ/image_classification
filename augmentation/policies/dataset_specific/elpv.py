from __future__ import annotations

import torchvision.transforms as transforms

from augmentation.policies.two_dim import defaults as defaults_2d


def build_train_transform(
    *,
    image_size: int = 300,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_test_transform(
    *,
    image_size: int = 300,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
