"""Augmentation registry.

Provides a single entry-point to build train/test transforms by dataset key.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

from augmentation.policies import (
    aid,
    chestxray14,
    cifar,
    diabetic_retina,
    fgvc_aircraft,
    flowers102,
    imagenet,
    neudet,
    pcam,
    resisc45,
    rxrx1,
    zooscannet,
)

TransformBuilder = Callable[..., object]


@dataclass(frozen=True)
class AugmentationSpec:
    build_train: TransformBuilder
    build_test: TransformBuilder
    default_kwargs: dict


def _filter_kwargs(builder: TransformBuilder, kwargs: dict) -> dict:
    """Pass only kwargs that a specific builder actually accepts."""
    signature = inspect.signature(builder)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    accepted = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in accepted}


_AUGMENTATION_REGISTRY: dict[str, AugmentationSpec] = {
    "cifar10": AugmentationSpec(
        build_train=cifar.build_train_transform,
        build_test=cifar.build_test_transform,
        default_kwargs={"cutout_size": 16},
    ),
    "cifar100": AugmentationSpec(
        build_train=cifar.build_train_transform,
        build_test=cifar.build_test_transform,
        default_kwargs={"cutout_size": 8},
    ),
    "imagenet": AugmentationSpec(
        build_train=imagenet.build_train_transform,
        build_test=imagenet.build_test_transform,
        default_kwargs={},
    ),
    "pcam": AugmentationSpec(
        build_train=pcam.build_train_transform,
        build_test=pcam.build_test_transform,
        default_kwargs={},
    ),
    "rxrx1": AugmentationSpec(
        build_train=rxrx1.build_train_transform,
        build_test=rxrx1.build_test_transform,
        default_kwargs={},
    ),
    "neudet": AugmentationSpec(
        build_train=neudet.build_train_transform,
        build_test=neudet.build_test_transform,
        default_kwargs={},
    ),
    "zooscannet": AugmentationSpec(
        build_train=zooscannet.build_train_transform,
        build_test=zooscannet.build_test_transform,
        default_kwargs={},
    ),
    "aid": AugmentationSpec(
        build_train=aid.build_train_transform,
        build_test=aid.build_test_transform,
        default_kwargs={},
    ),
    "chestxray14": AugmentationSpec(
        build_train=chestxray14.build_train_transform,
        build_test=chestxray14.build_test_transform,
        default_kwargs={},
    ),
    "resisc45": AugmentationSpec(
        build_train=resisc45.build_train_transform,
        build_test=resisc45.build_test_transform,
        default_kwargs={},
    ),
    "flowers102": AugmentationSpec(
        build_train=flowers102.build_train_transform,
        build_test=flowers102.build_test_transform,
        default_kwargs={},
    ),
    "fgvc_aircraft": AugmentationSpec(
        build_train=fgvc_aircraft.build_train_transform,
        build_test=fgvc_aircraft.build_test_transform,
        default_kwargs={},
    ),
    "diabetic_retina": AugmentationSpec(
        build_train=diabetic_retina.build_train_transform,
        build_test=diabetic_retina.build_test_transform,
        default_kwargs={},
    ),
}


def build_transforms(dataset: str, **overrides) -> tuple[object, object]:
    if dataset not in _AUGMENTATION_REGISTRY:
        raise ValueError(f"No augmentation registry entry for dataset={dataset!r}")
    spec = _AUGMENTATION_REGISTRY[dataset]
    kwargs = dict(spec.default_kwargs)
    kwargs.update(overrides)
    train_kwargs = _filter_kwargs(spec.build_train, kwargs)
    test_kwargs = _filter_kwargs(spec.build_test, kwargs)
    return spec.build_train(**train_kwargs), spec.build_test(**test_kwargs)
