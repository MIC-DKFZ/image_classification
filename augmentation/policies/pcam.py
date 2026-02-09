import torchvision.transforms as transforms
from .base_transform import BaseTransform

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TrainTransform(BaseTransform):
    """
    PCam training transforms.
    Rotation-invariant augmentations for histopathology patches.
    Native resolution is 96x96.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),  # Upscale from 96 to 256
                transforms.RandomCrop(224),  # Then crop to 224
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),  # Tissue has no preferred orientation
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(MEAN_IMGNET, STD_IMGNET),
            ]
        )
        return transform_train


class TrainTransformNative(BaseTransform):
    """
    PCam training transforms keeping native 96x96 resolution.
    Use this if your model supports smaller inputs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(MEAN_IMGNET, STD_IMGNET),
            ]
        )
        return transform_train


class TestTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(MEAN_IMGNET, STD_IMGNET),
            ]
        )
        return transform_test


class TestTransformNative(BaseTransform):
    """Native 96x96 resolution for test."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(MEAN_IMGNET, STD_IMGNET),
            ]
        )
        return transform_test


