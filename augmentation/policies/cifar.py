import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from randaugment import RandAugment, Cutout, CIFAR10Policy


def get_baseline(mean, std):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform_train


def get_baseline_cutout(mean, std, cutout_size):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        Cutout(size=cutout_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform_train


def get_auto_augmentation(mean, std, cutout_size):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        Cutout(size=cutout_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Random Erase with p=1 is an alternative to Cutout but worked slightly worse
        #transforms.RandomErasing(p=1,
        #                        scale=(0.125, 0.2), # range for how big the cutout should be compared to original img
        #                        ratio=(0.99, 1.0), # squares
        #                        value=0, inplace=False)
    ])

    return transform_train


def get_rand_augmentation(mean, std, cutout_size):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugment(),
        Cutout(size=cutout_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Random Erase with p=1 is an alternative to Cutout but worked slightly worse
        #transforms.RandomErasing(p=1,
        #                         scale=(0.125, 0.2),
        #                         ratio=(0.99, 1.0),
        #                         value=0, inplace=False)
    ])

    return transform_train


def get_album(mean, std):
    # custom albumentations policy
    transform_train = A.Compose([
        A.InvertImg(always_apply=False, p=0.2),
        A.PadIfNeeded(always_apply=False, p=0.2, min_height=36, min_width=36, pad_height_divisor=None,
                      pad_width_divisor=None, border_mode=4, value=None, mask_value=None),
        A.RandomCrop(always_apply=1, p=0.2, height=32, width=32),
        A.HorizontalFlip(always_apply=False, p=0.2),
        A.RandomContrast(always_apply=False, p=0.2, limit=(-0.2, 0.2)),
        A.ShiftScaleRotate(always_apply=False, p=0.2, shift_limit_x=(-0.2, 0.2), shift_limit_y=(-0.2, 0.2),
                           scale_limit=(0.0, 0.0), rotate_limit=(0, 0), interpolation=1, border_mode=4, value=None,
                           mask_value=None),
        A.Rotate(always_apply=False, p=0.2, limit=(-30, 30), interpolation=1, border_mode=4, value=None,
                 mask_value=None),
        A.HorizontalFlip(always_apply=False, p=0.2),
        A.RandomBrightness(always_apply=False, p=0.2, limit=(-0.2, 0.2)),
        A.ShiftScaleRotate(always_apply=False, p=0.2, shift_limit_x=(0, 0), shift_limit_y=(0, 0),
                           scale_limit=(-0.19999999999999996, 0.19999999999999996), rotate_limit=(0, 0),
                           interpolation=1,
                           border_mode=4, value=None, mask_value=None),
        A.Equalize(always_apply=False, p=0.2, mode='cv', by_channels=True),
        A.InvertImg(always_apply=False, p=0.2),
        A.Rotate(always_apply=False, p=0.2, limit=(-30, 30), interpolation=1, border_mode=4, value=None,
                 mask_value=None),
        A.ShiftScaleRotate(always_apply=False, p=0.2, shift_limit_x=(-0.2, 0.2), shift_limit_y=(-0.2, 0.2),
                           scale_limit=(0.0, 0.0), rotate_limit=(0, 0), interpolation=1, border_mode=4, value=None,
                           mask_value=None),
        A.Equalize(always_apply=False, p=0.2, mode='cv', by_channels=True),
        A.RandomBrightness(always_apply=False, p=0.2, limit=(-0.2, 0.2)),
        A.ShiftScaleRotate(always_apply=False, p=0.2, shift_limit_x=(0, 0), shift_limit_y=(0, 0),
                           scale_limit=(-0.19999999999999996, 0.19999999999999996), rotate_limit=(0, 0),
                           interpolation=1,
                           border_mode=4, value=None, mask_value=None),
        A.Solarize(always_apply=False, p=0.2, threshold=(128, 128)),

        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return transform_train


def test_transform(mean, std):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform_test

