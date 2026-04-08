from __future__ import annotations

from typing import Sequence

from glovita.augmentation.policies.adapters import BatchgeneratorsTransformAdapter
from glovita.augmentation.policies.metadata import TrainPolicySpec


def _as_tuple3(patch_size: Sequence[int] | None) -> tuple[int, int, int]:
    if patch_size is None:
        return (128, 128, 128)
    patch_size = tuple(int(v) for v in patch_size)
    if len(patch_size) != 3:
        raise ValueError(f"Expected a 3D patch_size, got {patch_size!r}")
    return patch_size


def _compose(transforms):
    from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

    return BatchgeneratorsTransformAdapter(ComposeTransforms(transforms))


def _brightnessadditive_localgamma_transform_scale(x, y):
    import numpy as np

    return np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y])))


def _brightness_gradient_additive_max_strength(_x, _y):
    import numpy as np

    return np.random.uniform(-5, -1) if np.random.uniform() < 0.5 else np.random.uniform(1, 5)


def _local_gamma_gamma():
    import numpy as np

    return np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4)


def _spatial_transform(
    *,
    patch_size,
    rotation_range,
    scaling_range,
    p_rotation,
    p_scaling,
):
    from batchgeneratorsv2.helpers.scalar_type import RandomScalar
    from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

    return SpatialTransform(
        patch_size,
        patch_center_dist_from_border=0,
        random_crop=False,
        p_elastic_deform=0.0,
        p_rotation=p_rotation,
        rotation=RandomScalar(rotation_range),
        p_scaling=p_scaling,
        scaling=scaling_range,
        p_synchronize_scaling_across_axes=1,
        bg_style_seg_sampling=False,
    )


def build_level1_train_transform(*, patch_size: Sequence[int] | None = None):
    _ = _as_tuple3(patch_size)
    return _compose([])


def build_test_transform(*, patch_size: Sequence[int] | None = None):
    _ = _as_tuple3(patch_size)
    return _compose([])


def build_level2_train_transform(
    *,
    patch_size: Sequence[int] | None = None,
    mirror_axes: tuple[int, ...] = (0, 1, 2),
):
    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform

    patch_size = _as_tuple3(patch_size)
    transforms = [
        _spatial_transform(
            patch_size=patch_size,
            rotation_range=(-0.15, 0.15),
            scaling_range=(0.95, 1.05),
            p_rotation=0.2,
            p_scaling=0.15,
        )
    ]
    if mirror_axes:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))
    return _compose(transforms)


def build_level3_train_transform(
    *,
    patch_size: Sequence[int] | None = None,
    mirror_axes: tuple[int, ...] = (0, 1, 2),
):
    from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
    from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform

    patch_size = _as_tuple3(patch_size)
    transforms = [
        _spatial_transform(
            patch_size=patch_size,
            rotation_range=(-0.25, 0.25),
            scaling_range=(0.9, 1.1),
            p_rotation=0.25,
            p_scaling=0.2,
        ),
        RandomTransform(
            GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_channel=1, synchronize_channels=True),
            apply_probability=0.1,
        ),
        RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.85, 1.15)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=0.1,
        ),
    ]
    if mirror_axes:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))
    return _compose(transforms)


def build_level4_train_transform(
    *,
    patch_size: Sequence[int] | None = None,
    mirror_axes: tuple[int, ...] = (0, 1, 2),
):
    from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
    from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
    from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
    from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
    from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform

    patch_size = _as_tuple3(patch_size)
    transforms = [
        _spatial_transform(
            patch_size=patch_size,
            rotation_range=(-0.35, 0.35),
            scaling_range=(0.8, 1.2),
            p_rotation=0.3,
            p_scaling=0.25,
        ),
        RandomTransform(
            GaussianNoiseTransform(noise_variance=(0, 0.08), p_per_channel=1, synchronize_channels=True),
            apply_probability=0.12,
        ),
        RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True,
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.8, 1.2)),
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.8, 1.2)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.6, 1.0),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            ),
            apply_probability=0.2,
        ),
    ]
    if mirror_axes:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))
    return _compose(transforms)


def build_default_nnunet_train_transform(
    *,
    patch_size: Sequence[int] | None = None,
    mirror_axes: tuple[int, ...] = (0, 1, 2),
):
    from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
    from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
    from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
    from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
    from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
    from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform

    patch_size = _as_tuple3(patch_size)
    transforms = [
        _spatial_transform(
            patch_size=patch_size,
            rotation_range=(-0.5, 0.5),
            scaling_range=(0.7, 1.4),
            p_rotation=0.35,
            p_scaling=0.3,
        ),
        RandomTransform(
            GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
            apply_probability=0.15,
        ),
        RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.2),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True,
            ),
            apply_probability=0.2,
        ),
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1.0),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            ),
            apply_probability=0.25,
        ),
        RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=0.25,
        ),
        RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=0.1,
        ),
    ]
    if mirror_axes:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))
    return _compose(transforms)


def build_default_nnunet_da5_train_transform(
    *,
    patch_size: Sequence[int] | None = None,
    mirror_axes: tuple[int, ...] = (0, 1, 2),
):
    import numpy as np

    from batchgeneratorsv2.transforms.intensity.brightness import BrightnessAdditiveTransform
    from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
    from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
    from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
    from batchgeneratorsv2.transforms.local.brightness_gradient import (
        BrightnessGradientAdditiveTransform,
    )
    from batchgeneratorsv2.transforms.local.local_gamma import LocalGammaTransform
    from batchgeneratorsv2.transforms.noise.blank_rectangle import BlankRectangleTransform
    from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
    from batchgeneratorsv2.transforms.noise.median_filter import MedianFilterTransform
    from batchgeneratorsv2.transforms.noise.sharpen import SharpeningTransform
    from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
    from batchgeneratorsv2.transforms.spatial.rot90 import Rot90Transform
    from batchgeneratorsv2.transforms.spatial.transpose import TransposeAxesTransform
    from batchgeneratorsv2.transforms.utils.random import OneOfTransform, RandomTransform

    patch_size = _as_tuple3(patch_size)
    matching_axes = np.array([sum(i == j for j in patch_size) for i in patch_size])
    valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
    transforms = [
        _spatial_transform(
            patch_size=patch_size,
            rotation_range=(-0.5, 0.5),
            scaling_range=(0.7, 1.43),
            p_rotation=0.4,
            p_scaling=0.2,
        ),
    ]
    if np.any(matching_axes > 1):
        transforms.extend(
            [
                RandomTransform(
                    Rot90Transform(
                        num_axis_combinations=1,
                        num_rot_per_combination=(0, 1, 2, 3),
                        allowed_axes=set(valid_axes),
                    ),
                    apply_probability=0.5,
                ),
                RandomTransform(
                    TransposeAxesTransform(allowed_axes=set(valid_axes)),
                    apply_probability=0.5,
                ),
            ]
        )
    transforms.extend(
        [
            OneOfTransform(
                [
                    RandomTransform(
                        MedianFilterTransform((2, 8), p_same_for_each_channel=0, p_per_channel=0.5),
                        apply_probability=0.2,
                    ),
                    RandomTransform(
                        GaussianBlurTransform(
                            (0.3, 1.5),
                            synchronize_channels=False,
                            synchronize_axes=True,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.2,
                    ),
                ]
            ),
            RandomTransform(
                GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
                apply_probability=0.1,
            ),
            RandomTransform(
                BrightnessAdditiveTransform(mu=0, sigma=0.5, synchronize_channels=False, p_per_channel=0.5),
                apply_probability=0.1,
            ),
            OneOfTransform(
                [
                    RandomTransform(
                        ContrastTransform(
                            contrast_range=BGContrast((0.5, 2)),
                            preserve_range=True,
                            synchronize_channels=False,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.2,
                    ),
                    RandomTransform(
                        ContrastTransform(
                            contrast_range=BGContrast((0.5, 2)),
                            preserve_range=False,
                            synchronize_channels=False,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.2,
                    ),
                ]
            ),
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.25, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=None,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=0.15,
            ),
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.1,
            ),
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.1,
            ),
        ]
    )
    if mirror_axes:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))
    transforms.extend(
        [
            RandomTransform(
                BlankRectangleTransform(
                    rectangle_size=[[max(1, p // 10), p // 3] for p in patch_size],
                    rectangle_value=np.mean,
                    num_rectangles=(1, 5),
                    force_square=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.4,
            ),
            RandomTransform(
                BrightnessGradientAdditiveTransform(
                    scale=_brightnessadditive_localgamma_transform_scale,
                    loc=(-0.5, 1.5),
                    max_strength=_brightness_gradient_additive_max_strength,
                    mean_centered=False,
                    same_for_all_channels=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.3,
            ),
            RandomTransform(
                LocalGammaTransform(
                    scale=_brightnessadditive_localgamma_transform_scale,
                    loc=(-0.5, 1.5),
                    gamma=_local_gamma_gamma,
                    same_for_all_channels=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.3,
            ),
            RandomTransform(
                SharpeningTransform(
                    strength=(0.1, 1.0),
                    p_same_for_each_channel=0,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2,
            ),
        ]
    )
    return _compose(transforms)


TRAIN_POLICIES: dict[str, TrainPolicySpec] = {
    "default_3d_1": TrainPolicySpec(build_level1_train_transform),
    "default_3d_2": TrainPolicySpec(build_level2_train_transform),
    "default_3d_3": TrainPolicySpec(build_level3_train_transform),
    "default_3d_4": TrainPolicySpec(build_level4_train_transform),
    "default_nnunet": TrainPolicySpec(build_default_nnunet_train_transform),
    "default_nnunet_DA5": TrainPolicySpec(build_default_nnunet_da5_train_transform),
}

TEST_POLICIES: dict[str, object] = {
    "shared_default_3d": build_test_transform,
}
