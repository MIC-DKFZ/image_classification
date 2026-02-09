import torchvision.transforms as transforms
from .base_transform import BaseTransform

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TrainTransform(BaseTransform):
    """
    RxRx1 training transforms.
    Rotation-invariant augmentations for cell microscopy images.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),  # Ensure 256
                transforms.RandomCrop(224),  # 256 → 224
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),  # Cells have no preferred orientation
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
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


