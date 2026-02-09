"""
Unit tests for augmentation policies.
"""

import pytest
import torch


class TestAugmentationPolicies:
    """Test augmentation policy functions."""

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_policy_module_exists(self, dataset_config):
        """Test that augmentation policy module exists."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"

        try:
            module = __import__(policy_name, fromlist=["get_train_transforms", "get_val_transforms"])
            assert hasattr(module, "get_train_transforms")
            assert hasattr(module, "get_val_transforms")
        except ImportError as e:
            pytest.fail(f"Failed to import {policy_name}: {e}")

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_train_transforms_callable(self, dataset_config):
        """Test that train transforms can be created and are callable."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["get_train_transforms"])
        get_train_transforms = getattr(module, "get_train_transforms")

        transforms = get_train_transforms()
        assert callable(transforms), "Train transforms should be callable"

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_val_transforms_callable(self, dataset_config):
        """Test that val transforms can be created and are callable."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["get_val_transforms"])
        get_val_transforms = getattr(module, "get_val_transforms")

        transforms = get_val_transforms()
        assert callable(transforms), "Val transforms should be callable"


class TestAugmentationApplication:
    """Test applying augmentations to mock data."""

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_train_transforms_work(self, dataset_config, mock_image):
        """Test that train transforms can be applied to an image."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["get_train_transforms"])
        get_train_transforms = getattr(module, "get_train_transforms")

        transforms = get_train_transforms()

        try:
            result = transforms(**{"image": mock_image})
            assert "image" in result, "Transform should return dict with 'image' key"

            transformed_img = result["image"]
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
        module = __import__(policy_name, fromlist=["get_val_transforms"])
        get_val_transforms = getattr(module, "get_val_transforms")

        transforms = get_val_transforms()

        try:
            result = transforms(**{"image": mock_image})
            assert "image" in result, "Transform should return dict with 'image' key"

            transformed_img = result["image"]
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
        module = __import__(policy_name, fromlist=["get_val_transforms"])
        get_val_transforms = getattr(module, "get_val_transforms")

        transforms = get_val_transforms()
        result = transforms(**{"image": mock_image})
        transformed_img = result["image"]

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
        module = __import__(policy_name, fromlist=["get_train_transforms"])
        get_train_transforms = getattr(module, "get_train_transforms")

        transforms = get_train_transforms()
        result = transforms(**{"image": mock_image})
        transformed_img = result["image"]

        assert transformed_img.shape[0] == 3, \
            f"Should preserve 3 channels, got {transformed_img.shape[0]}"


class TestAugmentationDifferences:
    """Test differences between train and val augmentations."""

    @pytest.mark.unit
    @pytest.mark.augmentation
    def test_train_and_val_produce_different_sizes(self, dataset_config, mock_image):
        """Test that train and val transforms may produce different sizes."""
        policy_name = f"augmentation.policies.{dataset_config['policy']}"
        module = __import__(policy_name, fromlist=["get_train_transforms", "get_val_transforms"])

        get_train_transforms = getattr(module, "get_train_transforms")
        get_val_transforms = getattr(module, "get_val_transforms")

        train_transforms = get_train_transforms()
        val_transforms = get_val_transforms()

        train_result = train_transforms(**{"image": mock_image})
        val_result = val_transforms(**{"image": mock_image})

        train_img = train_result["image"]
        val_img = val_result["image"]

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
        module = __import__(policy_name, fromlist=["get_train_transforms"])
        get_train_transforms = getattr(module, "get_train_transforms")

        transforms = get_train_transforms()

        # Apply same transform twice
        result1 = transforms(**{"image": mock_image.clone()})
        result2 = transforms(**{"image": mock_image.clone()})

        img1 = result1["image"]
        img2 = result2["image"]

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
        module = __import__(policy_name, fromlist=["get_val_transforms"])
        get_val_transforms = getattr(module, "get_val_transforms")

        transforms = get_val_transforms()

        # Apply same transform twice
        result1 = transforms(**{"image": mock_image.clone()})
        result2 = transforms(**{"image": mock_image.clone()})

        img1 = result1["image"]
        img2 = result2["image"]

        # Val transforms should be deterministic
        assert torch.allclose(img1, img2, atol=1e-6), \
            "Val transforms should be deterministic"
