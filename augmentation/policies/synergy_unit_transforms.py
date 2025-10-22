import torchvision.transforms as transforms
from ..randaugment import RandAugment
from .base_transform import BaseTransform

MEAN_IMAGENET, STD_IMAGENET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MEAN_GOOGLE, STD_GOOGLE = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
MEAN_DATASET, STD_DATASET = (0.3448, 0.3807, 0.4082), (0.0910, 0.0650, 0.0552)
IMAGE_SIZE = 224


class FlipRotateTransform(BaseTransform):
    def __init__(self, image_size, norm, *args, **kwargs):
        super().__init__()
        self.image_size = image_size
        if norm == "imagenet":
            self.mean = MEAN_IMAGENET
            self.std = STD_IMAGENET
        elif norm == "google":
            self.mean = MEAN_GOOGLE
            self.std = STD_GOOGLE
        elif norm == "dataset":
            self.mean = MEAN_DATASET
            self.std = STD_DATASET


    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        return transform_train


class TestTransform(BaseTransform):
    def __init__(self, image_size, norm, *args, **kwargs):
        super().__init__()
        self.image_size = image_size
        if norm == "imagenet":
            self.mean = MEAN_IMAGENET
            self.std = STD_IMAGENET
        elif norm == "google":
            self.mean = MEAN_GOOGLE
            self.std = STD_GOOGLE
        elif norm == "dataset":
            self.mean = MEAN_DATASET
            self.std = STD_DATASET

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        return transform_train