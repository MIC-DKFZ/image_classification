import albumentations as A
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2

from ..randaugment import CIFAR10Policy, Cutout, RandAugment
from .base_transform import AlbumentationsTransformAdapter

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def build_baseline_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_baseline_cutout_transform(cutout_size: int):
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_autoaugment_transform(cutout_size: int):
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_train_transform(cutout_size: int = 16):
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_album_augment_transform():
    return AlbumentationsTransformAdapter(
        A.Compose(
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
    )


def build_test_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_test_transform_dino():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_train_transform_dino():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


# ---------------------------------------------------------------------------
# Backward compatibility wrappers
# ---------------------------------------------------------------------------


class BaselineTransform:
    def __call__(self):
        return build_baseline_transform()


class BaselineCutoutTransform:
    def __init__(self, cutout_size: int, *args, **kwargs):
        self.cutout_size = cutout_size

    def __call__(self):
        return build_baseline_cutout_transform(self.cutout_size)


class AutoAugmentTransform:
    def __init__(self, cutout_size: int, *args, **kwargs):
        self.cutout_size = cutout_size

    def __call__(self):
        return build_autoaugment_transform(self.cutout_size)


class RandAugmentTransform:
    def __init__(self, cutout_size: int = 16, *args, **kwargs):
        self.cutout_size = cutout_size

    def __call__(self):
        return build_train_transform(self.cutout_size)


class AlbumAugmentTransform:
    def __call__(self):
        return build_album_augment_transform()


class TestTransform:
    def __call__(self):
        return build_test_transform()


class TestTransform_dino:
    def __call__(self):
        return build_test_transform_dino()


class RandAugmentTransform_dino:
    def __call__(self):
        return build_train_transform_dino()


TrainTransform = RandAugmentTransform
