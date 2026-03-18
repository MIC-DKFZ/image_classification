import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from .base_transform import AlbumentationsTransformAdapter, BaseTransform

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MEAN, STD = (0.3448, 0.3807, 0.4082), (0.0910, 0.0650, 0.0552)


class FlipRotateTransformImgNetNorm(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        return AlbumentationsTransformAdapter(
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=640),
                    A.RandomCrop(height=600, width=600),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_REFLECT, p=0.5),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.0,
                        p=0.5,
                    ),
                    A.Resize(height=224, width=224),
                    A.Normalize(MEAN_IMGNET, STD_IMGNET),
                    ToTensorV2(),
                ]
            )
        )


class TestTransformImgNetNorm(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        return AlbumentationsTransformAdapter(
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=640),
                    A.CenterCrop(height=600, width=600),
                    A.Resize(height=224, width=224),
                    A.Normalize(MEAN_IMGNET, STD_IMGNET),
                    ToTensorV2(),
                ]
            )
        )
