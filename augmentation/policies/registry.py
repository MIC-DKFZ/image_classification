"""Augmentation registry.

User-facing selection happens through `src/configs/data.py`.
Policy metadata lives next to the actual augmentation implementation:

- shared 2D policies in `augmentation.policies.two_dim.defaults`
- shared 3D policies in `augmentation.policies.three_dim.defaults`
- dataset-specific custom policies in
  `augmentation.policies.dataset_specific.<dataset>`
"""
from __future__ import annotations

import inspect
from importlib import import_module

from augmentation.policies.metadata import TrainPolicySpec, TransformBuilder
from augmentation.policies.three_dim import defaults as defaults_3d
from augmentation.policies.two_dim import defaults as defaults_2d


_SHARED_TRAIN_POLICIES_BY_DIM: dict[int, dict[str, TrainPolicySpec]] = {
    2: defaults_2d.TRAIN_POLICIES,
    3: defaults_3d.TRAIN_POLICIES,
}

_SHARED_TEST_POLICIES_BY_DIM: dict[int, dict[str, TransformBuilder]] = {
    2: defaults_2d.TEST_POLICIES,
    3: defaults_3d.TEST_POLICIES,
}

_DATASET_POLICY_MODULES: dict[str, str] = {
    "cifar10": "cifar",
    "cifar100": "cifar",
}


def _filter_kwargs(builder: TransformBuilder, kwargs: dict) -> dict:
    signature = inspect.signature(builder)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    accepted = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in accepted}


def _compact_overrides(overrides: dict) -> dict:
    return {key: value for key, value in overrides.items() if value is not None}


def _load_dataset_policy_module(dataset: str):
    module_name = _DATASET_POLICY_MODULES.get(dataset, dataset)
    try:
        return import_module(f"augmentation.policies.dataset_specific.{module_name}")
    except ImportError as exc:
        raise ValueError(f"No augmentation policy module for dataset={dataset!r}") from exc


def _shared_train_policies(spatial_dim: int) -> dict[str, TrainPolicySpec]:
    try:
        return _SHARED_TRAIN_POLICIES_BY_DIM[spatial_dim]
    except KeyError as exc:
        raise ValueError(f"Unsupported augmentation spatial_dim={spatial_dim!r}") from exc


def _shared_test_policies(spatial_dim: int) -> dict[str, TransformBuilder]:
    try:
        return _SHARED_TEST_POLICIES_BY_DIM[spatial_dim]
    except KeyError as exc:
        raise ValueError(f"Unsupported augmentation spatial_dim={spatial_dim!r}") from exc


def list_available_policies(dataset: str) -> dict[str, list[str]]:
    """List train/test policy names available for one dataset."""
    module = _load_dataset_policy_module(dataset)
    spatial_dim = int(getattr(module, "SPATIAL_DIM"))
    train_policies = {
        **_shared_train_policies(spatial_dim),
        **getattr(module, "TRAIN_POLICIES", {}),
    }
    test_policies = {
        **_shared_test_policies(spatial_dim),
        **getattr(module, "TEST_POLICIES", {}),
    }
    return {
        "train": sorted(train_policies),
        "test": sorted(test_policies),
    }


def resolve_policy_names(
    dataset: str,
    train_policy: str | None = None,
    test_policy: str | None = None,
) -> tuple[str, str]:
    """Return effective train/test policy names.

    Dataset defaults now live in `src/configs/data.py`; the registry only checks
    and resolves policy availability.
    """
    if train_policy is None or test_policy is None:
        raise ValueError(
            f"Dataset {dataset!r} is missing explicit augmentation defaults. "
            "Set them in the dataset config under src/configs/data.py or override them via CLI."
        )
    return train_policy, test_policy


def _resolve_train_spec(dataset: str, policy_name: str) -> TrainPolicySpec:
    module = _load_dataset_policy_module(dataset)
    spatial_dim = int(getattr(module, "SPATIAL_DIM"))
    train_policies = {
        **_shared_train_policies(spatial_dim),
        **getattr(module, "TRAIN_POLICIES", {}),
    }
    if policy_name in train_policies:
        return train_policies[policy_name]
    raise ValueError(
        f"Unknown train augmentation policy {policy_name!r} for dataset={dataset!r}. "
        f"Known train policies: {sorted(train_policies)}."
    )


def _resolve_test_builder(dataset: str, policy_name: str) -> TransformBuilder:
    module = _load_dataset_policy_module(dataset)
    spatial_dim = int(getattr(module, "SPATIAL_DIM"))
    test_policies = {
        **_shared_test_policies(spatial_dim),
        **getattr(module, "TEST_POLICIES", {}),
    }
    if policy_name in test_policies:
        return test_policies[policy_name]
    raise ValueError(
        f"Unknown test augmentation policy {policy_name!r} for dataset={dataset!r}. "
        f"Known test policies: {sorted(test_policies)}."
    )


def build_transforms(
    dataset: str,
    train_policy: str | None = None,
    test_policy: str | None = None,
    **overrides,
) -> tuple[object, object]:
    """Build train and test transforms for one dataset.

    Train policies may carry dataset- or policy-specific default kwargs, while
    test policies receive only the effective explicit overrides.
    """
    resolved_train_policy, resolved_test_policy = resolve_policy_names(
        dataset,
        train_policy=train_policy,
        test_policy=test_policy,
    )
    train_spec = _resolve_train_spec(dataset, resolved_train_policy)
    test_builder = _resolve_test_builder(dataset, resolved_test_policy)

    train_kwargs = dict(train_spec.default_kwargs)
    train_kwargs.update(_compact_overrides(overrides))
    test_kwargs = _compact_overrides(overrides)

    filtered_train_kwargs = _filter_kwargs(train_spec.build, train_kwargs)
    filtered_test_kwargs = _filter_kwargs(test_builder, test_kwargs)
    return (
        train_spec.build(**filtered_train_kwargs),
        test_builder(**filtered_test_kwargs),
    )
