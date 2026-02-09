import torchvision.transforms as transforms
from .base_transform import BaseTransform

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TrainTransform(BaseTransform):
    """
    Diabetic Retinopathy training transforms.
    Rotation-invariant augmentations for fundus images.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),  # Fundus images are rotation-invariant
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle for medical
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


