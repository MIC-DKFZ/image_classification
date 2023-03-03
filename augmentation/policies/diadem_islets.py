from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
import numpy as np


def baseline_transform(mean, std):

    transform = Compose(
        [
            MeanStdNormalizationTransform(mean, std),
            NumpyToTensor(keys="data", cast_to="float")
        ]
    )

    return transform


def nnunetv2_lite_transform(mean, std):
    tr_transforms = []
    # tr_transforms.append(SpatialTransform(
    #     patch_size=None,
    #     do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    #     do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 0), angle_z=(0, 0),
    #     p_rot_per_axis=1,  # todo experiment with this
    #     do_scale=True, scale=(0.7, 1.4),
    #     border_mode_data="constant", border_cval_data=0, order_data=1,
    #     random_crop=False,
    #     p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    #     independent_scale_for_each_axis=False  # todo experiment with this
    # ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    # tr_transforms.append(MirrorTransform(axes=(0,), p_per_sample=0.5))

    MeanStdNormalizationTransform(mean, std),
    tr_transforms.append(NumpyToTensor(keys="data", cast_to="float"))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def test_transform(mean, std):

    transform_test = Compose(
        [
            MeanStdNormalizationTransform(mean, std),
            NumpyToTensor(keys="data", cast_to="float")
        ]
    )

    return transform_test
