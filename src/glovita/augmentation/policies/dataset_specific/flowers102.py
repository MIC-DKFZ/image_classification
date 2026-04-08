from __future__ import annotations

from glovita.augmentation.policies.metadata import TrainPolicySpec
from glovita.augmentation.policies.two_dim import defaults as defaults_2d


def build_train_transform(
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = defaults_2d.IMAGENET_MEAN,
    std: tuple[float, ...] = defaults_2d.IMAGENET_STD,
):
    _ = resize_size
    return defaults_2d.build_level3_train_transform(image_size=image_size, mean=mean, std=std)


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


SPATIAL_DIM = 2
TRAIN_POLICIES = {
    "dataset_specific": TrainPolicySpec(
        build_train_transform,
        {"image_size": 224, "resize_size": 256},
    ),
}
TEST_POLICIES: dict[str, object] = {}
