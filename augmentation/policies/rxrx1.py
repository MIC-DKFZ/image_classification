import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from .base_transform import AlbumentationsTransformAdapter, BaseTransform

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TrainTransform(BaseTransform):
    """
    RxRx1 training transforms.
    Rotation-invariant augmentations for cell microscopy images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        return AlbumentationsTransformAdapter(
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=256),
                    A.RandomCrop(height=224, width=224),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_REFLECT, p=0.5),
                    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0, p=0.5),
                    A.Normalize(MEAN_IMGNET, STD_IMGNET),
                    ToTensorV2(),
                ]
            )
        )


class TestTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        return AlbumentationsTransformAdapter(
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=256),
                    A.CenterCrop(height=224, width=224),
                    A.Normalize(MEAN_IMGNET, STD_IMGNET),
                    ToTensorV2(),
                ]
            )
        )
