import albumentations as A
from albumentations.pytorch import ToTensorV2

from .base_transform import AlbumentationsTransformAdapter, BaseTransform

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TrainTransform(BaseTransform):
    """
    Flowers-102 training transforms.
    Standard augmentations for fine-grained flower classification.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        return AlbumentationsTransformAdapter(
            A.Compose(
                [
                    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.0,
                        p=0.5,
                    ),
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
