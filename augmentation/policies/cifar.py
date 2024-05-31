from typing import Any, Callable

import albumentations as A
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2

from ..randaugment import CIFAR10Policy, Cutout, RandAugment
from .base_transform import BaseTransform

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


class BaselineTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(BaselineTransform, self).__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        return transform_train


class BaselineCutoutTransform(BaseTransform):
    def __init__(self, cutout_size: int, *args, **kwargs):
        super(BaselineCutoutTransform, self).__init__()
        self.cutout_size = cutout_size

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                Cutout(size=self.cutout_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        return transform_train


class AutoAugmentTransform(BaseTransform):
    def __init__(self, cutout_size: int, *args, **kwargs):
        super(AutoAugmentTransform, self).__init__()
        self.cutout_size = cutout_size

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                Cutout(size=self.cutout_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
                # Random Erase with p=1 is an alternative to Cutout but worked slightly worse
                # transforms.RandomErasing(p=1,
                #                        scale=(0.125, 0.2), # range for how big the cutout should be compared to original img
                #                        ratio=(0.99, 1.0), # squares
                #                        value=0, inplace=False)
            ]
        )

        return transform_train


class RandAugmentTransform(BaseTransform):
    def __init__(self, cutout_size: int, *args, **kwargs):
        super(RandAugmentTransform, self).__init__()
        self.cutout_size = cutout_size

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugment(),
                Cutout(size=self.cutout_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
                # Random Erase with p=1 is an alternative to Cutout but worked slightly worse
                # transforms.RandomErasing(p=1,
                #                        scale=(0.125, 0.2), # range for how big the cutout should be compared to original img
                #                        ratio=(0.99, 1.0), # squares
                #                        value=0, inplace=False)
            ]
        )

        return transform_train


class AlbumAugmentTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(AlbumAugmentTransform, self).__init__()

    def __call__(self):
        transform_train = A.Compose(
            [
                A.InvertImg(always_apply=False, p=0.2),
                A.PadIfNeeded(
                    always_apply=False,
                    p=0.2,
                    min_height=36,
                    min_width=36,
                    pad_height_divisor=None,
                    pad_width_divisor=None,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                ),
                A.RandomCrop(always_apply=1, p=0.2, height=32, width=32),
                A.HorizontalFlip(always_apply=False, p=0.2),
                A.RandomBrightnessContrast(always_apply=False, p=0.2),
                A.ShiftScaleRotate(
                    always_apply=False,
                    p=0.2,
                    shift_limit_x=(-0.2, 0.2),
                    shift_limit_y=(-0.2, 0.2),
                    scale_limit=(0.0, 0.0),
                    rotate_limit=(0, 0),
                    interpolation=1,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                ),
                A.Equalize(always_apply=False, p=0.2, mode="cv", by_channels=True),
                A.Solarize(always_apply=False, p=0.2, threshold=(128, 128)),
                A.Normalize(MEAN, STD),
                ToTensorV2(),
            ]
        )

        return transform_train


class TestTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(TestTransform, self).__init__()

    def __call__(self):
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        return transform_test
