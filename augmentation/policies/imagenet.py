import torchvision.transforms as transforms

from ..randaugment import Cutout, ImageNetPolicy, RandAugment
from .base_transform import BaseTransform
from torchvision.transforms import InterpolationMode
from timm.data import create_transform


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class BaselineTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(BaselineTransform, self).__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
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
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                Cutout(size=self.cutout_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        return transform_train


class AutoAugmentTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(AutoAugmentTransform, self).__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        return transform_train


class TimmRandAugmentTramsformOld(BaseTransform):
    def __init__(self, *args, **kwargs ):
        super().__init__()
        self.transform = create_transform(
                input_size=448,
                is_training=True,
                mean=MEAN,
                std=STD,
                interpolation='bicubic',
                auto_augment='rand-m9-mstd0.5',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1
            )
    
    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(448,
                    scale=(0.08, 1.0),
                    ratio=(3/4, 4/3),
                    interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                self.transform,
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        return transform_train


class TimmRandAugmentTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.transform = create_transform(
            input_size=448,
            is_training=True,
            mean=MEAN,
            std=STD,
            interpolation='bicubic',
            auto_augment='rand-m9-mstd0.5',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1
        )

    def __call__(self, img):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(448),
                self.transform,
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        return transform_train
        # img is PIL.Image or ndarray
        # return self.transform(img)


class RandAugmentTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(448),
                transforms.RandomHorizontalFlip(),
                RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        return transform_train


class TestTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_test = transforms.Compose(
            [
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        return transform_test


class RandAugmentTransform_224(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                # transforms.RandomResizedCrop(224),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        return transform_train


class TestTransform_224(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        return transform_test
