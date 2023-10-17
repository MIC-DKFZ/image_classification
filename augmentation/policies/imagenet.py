import torchvision.transforms as transforms

from ..randaugment import Cutout, ImageNetPolicy, RandAugment
from .base_transform import BaseTransform

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


class RandAugmentTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(RandAugmentTransform, self).__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        return transform_train


class TestTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(TestTransform, self).__init__()

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
