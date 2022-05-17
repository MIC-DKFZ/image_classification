import torchvision.transforms as transforms
from randaugment import RandAugment, Cutout, ImageNetPolicy


def get_baseline(mean, std):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform_train


def get_baseline_cutout(mean, std, cutout_size):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform_train


def get_auto_augmentation(mean, std):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            # Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform_train


def get_rand_augmentation(mean, std):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            # Cutout(size=cutout_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform_train


def test_transform(mean, std):

    transform_test = transforms.Compose(
        [
            # transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform_test
