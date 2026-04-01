import torchvision.transforms as transforms

MEAN_IMAGENET, STD_IMAGENET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MEAN_GOOGLE, STD_GOOGLE = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
MEAN_DATASET, STD_DATASET = (0.3448, 0.3807, 0.4082), (0.0910, 0.0650, 0.0552)
IMAGE_SIZE = 224


def _resolve_norm(norm: str):
    if norm == "imagenet":
        return MEAN_IMAGENET, STD_IMAGENET
    if norm == "google":
        return MEAN_GOOGLE, STD_GOOGLE
    if norm == "dataset":
        return MEAN_DATASET, STD_DATASET
    raise ValueError(f"Unsupported norm={norm!r}. Expected one of: imagenet, google, dataset")


def build_train_transform(image_size: int = IMAGE_SIZE, norm: str = "imagenet"):
    mean, std = _resolve_norm(norm)
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_test_transform(image_size: int = IMAGE_SIZE, norm: str = "imagenet"):
    mean, std = _resolve_norm(norm)
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class FlipRotateTransform:
    def __init__(self, image_size: int = IMAGE_SIZE, norm: str = "imagenet", *args, **kwargs):
        self.image_size = image_size
        self.norm = norm

    def __call__(self):
        return build_train_transform(self.image_size, self.norm)


class TestTransform:
    def __init__(self, image_size: int = IMAGE_SIZE, norm: str = "imagenet", *args, **kwargs):
        self.image_size = image_size
        self.norm = norm

    def __call__(self):
        return build_test_transform(self.image_size, self.norm)
