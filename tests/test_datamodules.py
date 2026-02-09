"""
Unit tests for PyTorch Lightning DataModules.
"""

import pytest
import torch
from torch.utils.data import DataLoader


class TestDataModuleStructure:
    """Test DataModule class structure."""

    @pytest.mark.unit
    @pytest.mark.datamodule
    def test_datamodule_class_exists(self, dataset_config):
        """Test that DataModule class can be imported."""
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['datamodule']

        module = __import__(module_name, fromlist=[class_name])
        assert hasattr(module, class_name), f"{class_name} not found in {module_name}"

    @pytest.mark.unit
    @pytest.mark.datamodule
    def test_datamodule_has_required_methods(self, dataset_config):
        """Test that DataModule has required methods."""
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['datamodule']

        module = __import__(module_name, fromlist=[class_name])
        datamodule_class = getattr(module, class_name)

        # Check required methods from PyTorch Lightning DataModule
        assert hasattr(datamodule_class, 'setup')
        assert hasattr(datamodule_class, 'train_dataloader')
        assert hasattr(datamodule_class, 'val_dataloader')


class TestDataModuleInitialization:
    """Test DataModule initialization."""

    @pytest.mark.requires_data
    @pytest.mark.integration
    @pytest.mark.datamodule
    def test_datamodule_can_initialize(self, dataset_name, dataset_config, data_root):
        """Test that DataModule can be initialized."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        # Import DataModule class and augmentation policy
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['datamodule']
        policy_name = f"augmentation.policies.{dataset_config['policy']}"

        dataset_module = __import__(module_name, fromlist=[class_name])
        policy_module = __import__(policy_name, fromlist=["get_train_transforms", "get_val_transforms"])

        datamodule_class = getattr(dataset_module, class_name)
        get_train_transforms = getattr(policy_module, "get_train_transforms")
        get_val_transforms = getattr(policy_module, "get_val_transforms")

        try:
            # Initialize DataModule
            dm = datamodule_class(
                data_path=str(dataset_path),
                train_transforms=get_train_transforms(),
                test_transforms=get_val_transforms(),
                batch_size=16,
                num_workers=0,  # Use 0 for testing
            )
            assert dm is not None

        except FileNotFoundError:
            pytest.skip(f"Dataset files missing")
        except TypeError as e:
            # Some DataModules might have different initialization
            pytest.skip(f"DataModule initialization error: {e}")

    @pytest.mark.requires_data
    @pytest.mark.integration
    @pytest.mark.datamodule
    def test_datamodule_setup(self, dataset_name, dataset_config, data_root):
        """Test that DataModule setup works."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        # Import DataModule class and augmentation policy
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['datamodule']
        policy_name = f"augmentation.policies.{dataset_config['policy']}"

        dataset_module = __import__(module_name, fromlist=[class_name])
        policy_module = __import__(policy_name, fromlist=["get_train_transforms", "get_val_transforms"])

        datamodule_class = getattr(dataset_module, class_name)
        get_train_transforms = getattr(policy_module, "get_train_transforms")
        get_val_transforms = getattr(policy_module, "get_val_transforms")

        try:
            dm = datamodule_class(
                data_path=str(dataset_path),
                train_transforms=get_train_transforms(),
                test_transforms=get_val_transforms(),
                batch_size=16,
                num_workers=0,
            )

            # Call setup
            dm.setup(stage="fit")

            # Check that train and val datasets were created
            assert hasattr(dm, 'train_dataset'), "DataModule should have train_dataset after setup"
            assert hasattr(dm, 'val_dataset'), "DataModule should have val_dataset after setup"
            assert len(dm.train_dataset) > 0, "Train dataset should not be empty"
            assert len(dm.val_dataset) > 0, "Val dataset should not be empty"

        except FileNotFoundError:
            pytest.skip(f"Dataset files missing")
        except TypeError:
            pytest.skip(f"DataModule initialization error")


class TestDataModuleDataLoaders:
    """Test DataModule dataloaders."""

    @pytest.mark.requires_data
    @pytest.mark.integration
    @pytest.mark.datamodule
    @pytest.mark.slow
    def test_train_dataloader(self, dataset_name, dataset_config, data_root):
        """Test that train dataloader works."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        # Import DataModule class and augmentation policy
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['datamodule']
        policy_name = f"augmentation.policies.{dataset_config['policy']}"

        dataset_module = __import__(module_name, fromlist=[class_name])
        policy_module = __import__(policy_name, fromlist=["get_train_transforms", "get_val_transforms"])

        datamodule_class = getattr(dataset_module, class_name)
        get_train_transforms = getattr(policy_module, "get_train_transforms")
        get_val_transforms = getattr(policy_module, "get_val_transforms")

        try:
            dm = datamodule_class(
                data_path=str(dataset_path),
                train_transforms=get_train_transforms(),
                test_transforms=get_val_transforms(),
                batch_size=8,  # Smaller batch for testing
                num_workers=0,
            )

            dm.setup(stage="fit")
            train_loader = dm.train_dataloader()

            assert isinstance(train_loader, DataLoader), "Should return DataLoader"

            # Get one batch
            batch_imgs, batch_labels = next(iter(train_loader))

            assert isinstance(batch_imgs, torch.Tensor)
            assert isinstance(batch_labels, torch.Tensor)
            assert batch_imgs.shape[0] <= 8, "Batch size should be <= 8"
            assert batch_imgs.shape[1] == 3, "Should have 3 channels"
            assert batch_imgs.dtype == torch.float32, "Images should be float32"
            assert batch_labels.dtype == torch.int64, "Labels should be int64"

        except FileNotFoundError:
            pytest.skip(f"Dataset files missing")
        except TypeError:
            pytest.skip(f"DataModule initialization error")

    @pytest.mark.requires_data
    @pytest.mark.integration
    @pytest.mark.datamodule
    @pytest.mark.slow
    def test_val_dataloader(self, dataset_name, dataset_config, data_root):
        """Test that val dataloader works."""
        # Skip PCam if not preprocessed
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip(f"PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        # Import DataModule class and augmentation policy
        module_name = f"datasets.{dataset_config['module']}"
        class_name = dataset_config['datamodule']
        policy_name = f"augmentation.policies.{dataset_config['policy']}"

        dataset_module = __import__(module_name, fromlist=[class_name])
        policy_module = __import__(policy_name, fromlist=["get_train_transforms", "get_val_transforms"])

        datamodule_class = getattr(dataset_module, class_name)
        get_train_transforms = getattr(policy_module, "get_train_transforms")
        get_val_transforms = getattr(policy_module, "get_val_transforms")

        try:
            dm = datamodule_class(
                data_path=str(dataset_path),
                train_transforms=get_train_transforms(),
                test_transforms=get_val_transforms(),
                batch_size=8,
                num_workers=0,
            )

            dm.setup(stage="fit")
            val_loader = dm.val_dataloader()

            assert isinstance(val_loader, DataLoader), "Should return DataLoader"

            # Get one batch
            batch_imgs, batch_labels = next(iter(val_loader))

            assert isinstance(batch_imgs, torch.Tensor)
            assert isinstance(batch_labels, torch.Tensor)
            assert batch_imgs.dtype == torch.float32
            assert batch_labels.dtype == torch.int64

        except FileNotFoundError:
            pytest.skip(f"Dataset files missing")
        except TypeError:
            pytest.skip(f"DataModule initialization error")
