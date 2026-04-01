import torchvision.transforms as transforms

MEAN_IMGNET, STD_IMGNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_train_transform():
    return transforms.Compose(
        [
            transforms.Resize(300),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_IMGNET, STD_IMGNET),
        ]
    )


def build_test_transform():
    return transforms.Compose(
        [
            transforms.Resize(300),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_IMGNET, STD_IMGNET),
        ]
    )


class FlipRotateTransformImgNetNorm:
    def __call__(self):
        return build_train_transform()


class TestTransformImgNetNorm:
    def __call__(self):
        return build_test_transform()
