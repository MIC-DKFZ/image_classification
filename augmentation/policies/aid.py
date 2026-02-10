import torchvision.transforms as transforms
from ..randaugment import RandAugment
from .base_transform import BaseTransform

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MEAN, STD = (0.3448, 0.3807, 0.4082), (0.0910, 0.0650, 0.0552)


class FlipRotateTransformImgNetNorm(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.Resize(640),
                transforms.RandomCrop(600),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(MEAN_IMGNET, STD_IMGNET),
            ]
        )
        return transform_train


class TestTransformImgNetNorm(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.Resize(640),
                transforms.CenterCrop(600),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(MEAN_IMGNET, STD_IMGNET),
            ]
        )
        return transform_train


