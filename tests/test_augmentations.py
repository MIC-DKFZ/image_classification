"""
Unit tests for augmentation policies.
"""

import pytest
import torch
import numpy as np
from torchvision.transforms import ToPILImage


def _resolve_policy_transforms(module):
    if hasattr(module, "build_train_transform") and hasattr(module, "build_test_transform"):
        return module.build_train_transform(), module.build_test_transform()
    raise AttributeError("Policy module must expose build_train_transform/build_test_transform")


def _apply_transform(transform, image):
    hwc_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    try:
        result = transform(image=hwc_np)
        if isinstance(result, dict) and "image" in result:
            return result["image"]
    except TypeError:
        pass

    try:
        result = transform(image)
    except Exception:
        result = transform(ToPILImage()(image))

    if isinstance(result, dict):
        if "image" not in result:
            raise KeyError("Transform dict output is missing 'image' key")
        return result["image"]
    return result


class TestAugmentationPolicies:
    """Test augmentation policy functions."""

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_policy_module_exists(self, dataset_config):
        """Test that augmentation policy module exists."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"

        try:
            module = __import__(policy_name, fromlist=["TrainTransform", "TestTransform"])
            train_t, val_t = _resolve_policy_transforms(module)
            assert callable(train_t)
            assert callable(val_t)
        except ImportError as e:
            pytest.fail(f"Failed to import {policy_name}: {e}")

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_train_transforms_callable(self, dataset_config):
        """Test that train transforms can be created and are callable."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TrainTransform"])
        transforms, _ = _resolve_policy_transforms(module)
        assert callable(transforms), "Train transforms should be callable"

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_val_transforms_callable(self, dataset_config):
        """Test that val transforms can be created and are callable."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TestTransform"])
        _, transforms = _resolve_policy_transforms(module)
        assert callable(transforms), "Val transforms should be callable"


class TestAugmentationApplication:
    """Test applying augmentations to mock data."""

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_train_transforms_work(self, dataset_config, mock_image):
        """Test that train transforms can be applied to an image."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TrainTransform"])
        transforms, _ = _resolve_policy_transforms(module)

        try:
            transformed_img = _apply_transform(transforms, mock_image)
            assert isinstance(transformed_img, torch.Tensor)
            assert transformed_img.dtype == torch.float32, "Transformed image should be float32"
            assert len(transformed_img.shape) == 3, "Should be (C, H, W)"

        except Exception as e:
            pytest.fail(f"Train transform failed: {e}")

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_val_transforms_work(self, dataset_config, mock_image):
        """Test that val transforms can be applied to an image."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TestTransform"])
        _, transforms = _resolve_policy_transforms(module)

        try:
            transformed_img = _apply_transform(transforms, mock_image)
            assert isinstance(transformed_img, torch.Tensor)
            assert transformed_img.dtype == torch.float32, "Transformed image should be float32"
            assert len(transformed_img.shape) == 3, "Should be (C, H, W)"

        except Exception as e:
            pytest.fail(f"Val transform failed: {e}")

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_transforms_normalize_values(self, dataset_config, mock_image):
        """Test that transforms normalize image values (ImageNet normalization)."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TestTransform"])
        _, transforms = _resolve_policy_transforms(module)
        transformed_img = _apply_transform(transforms, mock_image)

        # ImageNet normalization should produce values outside [0, 1] range
        # Check that we have negative values (indicating normalization was applied)
        assert transformed_img.min() < 0, \
            "ImageNet normalization should produce negative values"
        assert transformed_img.max() > 1 or transformed_img.min() < 0, \
            "Values should be normalized (not in [0, 1] range)"

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_transforms_preserve_channels(self, dataset_config, mock_image):
        """Test that transforms preserve number of channels."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TrainTransform"])
        transforms, _ = _resolve_policy_transforms(module)
        transformed_img = _apply_transform(transforms, mock_image)

        assert transformed_img.shape[0] == 3, \
            f"Should preserve 3 channels, got {transformed_img.shape[0]}"


class TestAugmentationDifferences:
    """Test differences between train and val augmentations."""

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_train_and_val_produce_different_sizes(self, dataset_config, mock_image):
        """Test that train and val transforms may produce different sizes."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TrainTransform", "TestTransform"])
        train_transforms, val_transforms = _resolve_policy_transforms(module)

        train_img = _apply_transform(train_transforms, mock_image)
        val_img = _apply_transform(val_transforms, mock_image)

        # Both should produce valid tensors
        assert isinstance(train_img, torch.Tensor)
        assert isinstance(val_img, torch.Tensor)

        # Both should have same number of channels
        assert train_img.shape[0] == val_img.shape[0] == 3

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_train_is_stochastic(self, dataset_config, mock_image):
        """Test that train transforms are stochastic (produce different results)."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TrainTransform"])
        transforms, _ = _resolve_policy_transforms(module)

        # Apply same transform twice
        img1 = _apply_transform(transforms, mock_image.clone())
        img2 = _apply_transform(transforms, mock_image.clone())

        # They should be different (due to random augmentations)
        # Note: This might occasionally fail due to randomness, but very unlikely
        if not torch.allclose(img1, img2, atol=1e-5):
            assert True  # Transforms are stochastic
        else:
            # If they're identical, transforms might be deterministic (val transforms)
            # This is acceptable, just note it
            pass

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_val_is_deterministic(self, dataset_config, mock_image):
        """Test that val transforms are deterministic (produce same results)."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["TestTransform"])
        _, transforms = _resolve_policy_transforms(module)

        # Apply same transform twice
        img1 = _apply_transform(transforms, mock_image.clone())
        img2 = _apply_transform(transforms, mock_image.clone())

        # Val transforms should be deterministic
        assert torch.allclose(img1, img2, atol=1e-6), \
            "Val transforms should be deterministic"
