"""
Unit tests for dataset classes.
"""

import pytest
import torch
from pathlib import Path


class TestDatasetStructure:
    """Test dataset class structure and interface."""

    @pytest.mark.unit
    def test_dataset_class_exists(self, dataset_config):
        """Test that dataset class can be imported."""
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['class']

        module = __import__(module_name, fromlist=[class_name])
        assert hasattr(module, class_name), f"{class_name} not found in {module_name}"

    @pytest.mark.unit
    def test_dataset_has_required_methods(self, dataset_config):
        """Test that dataset has required methods."""
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['class']

        module = __import__(module_name, fromlist=[class_name])
        dataset_class = getattr(module, class_name)

        # Check required methods
        assert hasattr(dataset_class, '__init__')
        assert hasattr(dataset_class, '__len__')
        assert hasattr(dataset_class, '__getitem__')


class TestDatasetLoading:
    """Test dataset loading with actual data."""

    @pytest.mark.requires_data
    @pytest.mark.integration
    def test_dataset_can_load(self, dataset_name, dataset_config, data_root):
        """Test that dataset can be loaded with actual data."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction. Run pcam_split.py first.")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found at {dataset_path}")

        # Import dataset class
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['class']
        module = __import__(module_name, fromlist=[class_name])
        dataset_class = getattr(module, class_name)

        # Try to load dataset
        try:
            dataset = dataset_class(
                root=str(dataset_path),
                split="train",
                transform=None
            )
            assert len(dataset) > 0, f"{dataset_name} has no samples"
        except FileNotFoundError as e:
            pytest.skip(f"Dataset files missing: {e}")

    @pytest.mark.requires_data
    @pytest.mark.integration
    def test_dataset_getitem(self, dataset_name, dataset_config, data_root):
        """Test that dataset __getitem__ works."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        # Import dataset class
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['class']
        module = __import__(module_name, fromlist=[class_name])
        dataset_class = getattr(module, class_name)

        try:
            dataset = dataset_class(
                root=str(dataset_path),
                split="train",
                transform=None
            )

            # Get first item
            img, label = dataset[0]

            # Check types
            assert isinstance(img, torch.Tensor), f"Image should be tensor, got {type(img)}"
            assert isinstance(label, (int, torch.Tensor)), f"Label should be int or tensor"

            # Check image shape (C, H, W)
            assert len(img.shape) == 3, f"Image should be (C, H, W), got {img.shape}"
            assert img.shape[0] == 3, f"Image should have 3 channels, got {img.shape[0]}"

            # Check label range
            if isinstance(label, torch.Tensor):
                label = label.item()
            assert 0 <= label < dataset_config['num_classes'], \
                f"Label {label} out of range [0, {dataset_config['num_classes']})"

        except FileNotFoundError:
            pytest.skip(f"Dataset files missing")

    @pytest.mark.requires_data
    @pytest.mark.integration
    def test_dataset_with_transforms(self, dataset_name, dataset_config, data_root):
        """Test dataset with augmentation transforms."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        # Import dataset class and transforms
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['class']
        policy_name = f"augmentation.policies.{dataset_config['policy']}"

        dataset_module = __import__(module_name, fromlist=[class_name])
        policy_module = __import__(policy_name, fromlist=["get_train_transforms"])

        dataset_class = getattr(dataset_module, class_name)
        get_train_transforms = getattr(policy_module, "get_train_transforms")

        try:
            transforms = get_train_transforms()
            dataset = dataset_class(
                root=str(dataset_path),
                split="train",
                transform=transforms
            )

            # Get first item with transforms
            img, label = dataset[0]

            # Check image is normalized (values should be outside [0, 1] range)
            assert img.dtype == torch.float32, f"Transformed image should be float32"
            assert img.min() < 0 or img.max() > 1, \
                "Image should be normalized (ImageNet normalization gives negative values)"

        except FileNotFoundError:
            pytest.skip(f"Dataset files missing")


class TestDatasetSplits:
    """Test dataset splits are valid."""

    @pytest.mark.requires_data
    @pytest.mark.integration
    def test_all_splits_exist(self, dataset_name, dataset_config, data_root):
        """Test that train, val, test splits all exist."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction")

        # Skip DiabeticRetinopathy (uses different split format)
        if dataset_name == "DiabeticRetinopathy":
            pytest.skip(f"{dataset_name} uses custom split format")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        # Import dataset class
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['class']
        module = __import__(module_name, fromlist=[class_name])
        dataset_class = getattr(module, class_name)

        try:
            for split in ["train", "val", "test"]:
                dataset = dataset_class(
                    root=str(dataset_path),
                    split=split,
                    transform=None
                )
                assert len(dataset) > 0, f"{dataset_name} {split} split is empty"

        except FileNotFoundError:
            pytest.skip(f"Dataset files missing")

    @pytest.mark.requires_data
    @pytest.mark.integration
    def test_splits_no_overlap(self, dataset_name, dataset_config, data_root):
        """Test that train/val/test splits don't overlap."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction")

        # Skip DiabeticRetinopathy (uses different split format)
        if dataset_name == "DiabeticRetinopathy":
            pytest.skip(f"{dataset_name} uses custom split format")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        splits_file = dataset_path / "splits.json"
        if not splits_file.exists():
            pytest.skip(f"splits.json not found")

        import json
        with open(splits_file) as f:
            splits = json.load(f)

        train_ids = set(splits["train"])
        val_ids = set(splits["val"])
        test_ids = set(splits["test"])

        # Check no overlap
        assert train_ids.isdisjoint(val_ids), "Train and val splits overlap"
        assert train_ids.isdisjoint(test_ids), "Train and test splits overlap"
        assert val_ids.isdisjoint(test_ids), "Val and test splits overlap"
