import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from .base_transform import AlbumentationsTransformAdapter

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_train_transform():
    return AlbumentationsTransformAdapter(
        A.Compose(
            [
                A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.0,
                    hue=0.0,
                    p=0.5,
                ),
                A.Normalize(MEAN_IMGNET, STD_IMGNET),
                ToTensorV2(),
            ]
        )
    )


def build_test_transform():
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


class TrainTransform:
    def __call__(self):
        return build_train_transform()


class TestTransform:
    def __call__(self):
        return build_test_transform()
