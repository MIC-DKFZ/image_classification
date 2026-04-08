"""Tests for dataset factory / dataloader wiring."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from glovita.datasets.factory import _DATASET_REGISTRY, build_dataloaders
from glovita.configs import data as data_cfg_module
from glovita.configs.dataloading import DataloadingConfig


def _build_data_config(dataset_config, dataset_path: Path):
    config_cls = getattr(data_cfg_module, dataset_config["config_class"])
    return config_cls(
        data_root_dir=dataset_path,
        data_fraction=1.0,
    ), DataloadingConfig(batch_size=8, eval_batch_size=8, num_workers=0)


class TestDatasetFactoryStructure:
    @pytest.mark.unit
    @pytest.mark.datamodule
    def test_dataset_registered(self, dataset_config):
        dataset_key = dataset_config["dataset_key"]
        assert dataset_key in _DATASET_REGISTRY

    @pytest.mark.unit
    @pytest.mark.datamodule
    def test_dataset_config_class_exists(self, dataset_config):
        assert hasattr(data_cfg_module, dataset_config["config_class"])


class TestDatasetFactoryIntegration:
    @pytest.mark.requires_data
    @pytest.mark.integration
    @pytest.mark.datamodule
    def test_factory_builds_dataloaders(self, dataset_name, dataset_config, data_root):
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip("PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        cfg, dataloading = _build_data_config(dataset_config, dataset_path)
        train_loader, val_loader, test_loader = build_dataloaders(cfg, dataloading)

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    @pytest.mark.requires_data
    @pytest.mark.integration
    @pytest.mark.datamodule
    @pytest.mark.slow
    def test_factory_train_batch_shape(self, dataset_name, dataset_config, data_root):
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip("PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        cfg, dataloading = _build_data_config(dataset_config, dataset_path)
        train_loader, _, _ = build_dataloaders(cfg, dataloading)
        images, labels = next(iter(train_loader))

        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape[0] <= dataloading.batch_size
        assert images.shape[1] == 3
        assert images.dtype == torch.float32

    @pytest.mark.requires_data
    @pytest.mark.integration
    @pytest.mark.datamodule
    @pytest.mark.slow
    def test_factory_val_batch_shape(self, dataset_name, dataset_config, data_root):
        if dataset_name == "PCam":
            pcam_images = data_root / dataset_name / "images"
            if not pcam_images.exists():
                pytest.skip("PCam requires H5 extraction")

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")

        cfg, dataloading = _build_data_config(dataset_config, dataset_path)
        _, val_loader, _ = build_dataloaders(cfg, dataloading)
        images, labels = next(iter(val_loader))

        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape[1] == 3
        assert images.dtype == torch.float32
