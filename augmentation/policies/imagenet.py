import torchvision.transforms as transforms
from timm.data import create_transform
from torchvision.transforms import InterpolationMode

from ..randaugment import Cutout, ImageNetPolicy, RandAugment

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_baseline_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_baseline_cutout_transform(cutout_size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_autoaugment_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_timm_randaugment_transform():
    return create_transform(
        input_size=448,
        is_training=True,
        mean=MEAN,
        std=STD,
        interpolation="bicubic",
        scale=(0.08, 1.0),
        ratio=(3 / 4, 4 / 3),
        auto_augment="rand-m9-mstd0.5",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )


def build_timm_randaugment_transform_old():
    timm_transform = create_transform(
        input_size=448,
        is_training=True,
        mean=MEAN,
        std=STD,
        interpolation="bicubic",
        auto_augment="rand-m9-mstd0.5",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                448,
                scale=(0.08, 1.0),
                ratio=(3 / 4, 4 / 3),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            timm_transform,
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(448),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_test_transform():
    return transforms.Compose(
        [
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_train_transform_224():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_test_transform_224():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


class BaselineTransform:
    def __call__(self):
        return build_baseline_transform()


class BaselineCutoutTransform:
    def __init__(self, cutout_size: int, *args, **kwargs):
        self.cutout_size = cutout_size

    def __call__(self):
        return build_baseline_cutout_transform(self.cutout_size)


class AutoAugmentTransform:
    def __call__(self):
        return build_autoaugment_transform()


class TimmRandAugmentTramsformOld:
    def __call__(self):
        return build_timm_randaugment_transform_old()


class TimmRandAugmentTransform:
    def __call__(self):
        return build_timm_randaugment_transform()


class RandAugmentTransform:
    def __call__(self):
        return build_train_transform()


class TestTransform:
    def __call__(self):
        return build_test_transform()


class RandAugmentTransform_224:
    def __call__(self):
        return build_train_transform_224()


class TestTransform_224:
    def __call__(self):
        return build_test_transform_224()
