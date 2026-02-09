"""
Pytest configuration and shared fixtures.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np


# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Set up environment variables for all tests."""
    if "DATA_ROOT" not in os.environ:
        os.environ["DATA_ROOT"] = "/home/d246a/Documents/data/SynergyUnitDatasets"
    yield


@pytest.fixture
def data_root():
    """Get the DATA_ROOT path."""
    return Path(os.environ.get("DATA_ROOT", "/home/d246a/Documents/data/SynergyUnitDatasets"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Dataset configuration fixtures
@pytest.fixture(params=[
    "AID",
    "ZooScanNet",
    "ChestXray14",
    "NEUDET",
    "RxRx1",
    "Flowers102",
    "RESISC45",
    "PCam",
    "DiabeticRetinopathy",
    "FGVCAircraft",
])
def dataset_name(request):
    """Parametrized fixture for all dataset names."""
    return request.param


@pytest.fixture
def dataset_config(dataset_name):
    """Get configuration for a specific dataset."""
    configs = {
        "AID": {
            "module": "aid",
            "class": "AIDData",
            "datamodule": "AIDDataModule",
            "policy": "aid",
            "num_classes": 30,
            "expected_shape": (3, 224, 224),
        },
        "ZooScanNet": {
            "module": "zooscannet",
            "class": "ZooScanNetData",
            "datamodule": "ZooScanNetDataModule",
            "policy": "zooscannet",
            "num_classes": 116,
            "expected_shape": (3, 224, 224),
        },
        "ChestXray14": {
            "module": "chestxray14",
            "class": "ChestXray14Data",
            "datamodule": "ChestXray14DataModule",
            "policy": "chestxray14",
            "num_classes": 15,
            "expected_shape": (3, 224, 224),
        },
        "NEUDET": {
            "module": "neudet",
            "class": "NEUDETData",
            "datamodule": "NEUDETDataModule",
            "policy": "neudet",
            "num_classes": 6,
            "expected_shape": (3, 224, 224),
        },
        "RxRx1": {
            "module": "rxrx1",
            "class": "RxRx1Data",
            "datamodule": "RxRx1DataModule",
            "policy": "rxrx1",
            "num_classes": 1139,
            "expected_shape": (3, 224, 224),
        },
        "Flowers102": {
            "module": "flowers102",
            "class": "Flowers102Data",
            "datamodule": "Flowers102DataModule",
            "policy": "flowers102",
            "num_classes": 102,
            "expected_shape": (3, 224, 224),
        },
        "RESISC45": {
            "module": "resisc45",
            "class": "RESISC45Data",
            "datamodule": "RESISC45DataModule",
            "policy": "resisc45",
            "num_classes": 45,
            "expected_shape": (3, 224, 224),
        },
        "PCam": {
            "module": "pcam",
            "class": "PCamData",
            "datamodule": "PCamDataModule",
            "policy": "pcam",
            "num_classes": 2,
            "expected_shape": (3, 224, 224),
        },
        "DiabeticRetinopathy": {
            "module": "diabetic_retina",
            "class": "EyePACSData",
            "datamodule": "EyePACSDataModule",
            "policy": "diabetic_retina",
            "num_classes": 5,
            "expected_shape": (3, 224, 224),
        },
        "FGVCAircraft": {
            "module": "fgvc_aircraft",
            "class": "FGVCAircraftData",
            "datamodule": "FGVCAircraftDataModule",
            "policy": "fgvc_aircraft",
            "num_classes": 100,
            "expected_shape": (3, 224, 224),
        },
    }
    return configs[dataset_name]


# Mock data fixtures
@pytest.fixture
def mock_image():
    """Create a mock RGB image tensor."""
    return torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)


@pytest.fixture
def mock_batch():
    """Create a mock batch of images."""
    batch_size = 16
    images = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.uint8)
    labels = torch.randint(0, 10, (batch_size,), dtype=torch.int64)
    return images, labels


@pytest.fixture
def mock_normalized_image():
    """Create a mock normalized image tensor."""
    return torch.randn(3, 224, 224, dtype=torch.float32)


# Augmentation fixtures
@pytest.fixture(params=["train", "val"])
def transform_type(request):
    """Parametrized fixture for transform types."""
    return request.param


# Skip markers for datasets that need preprocessing
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark tests that require actual data
        if "requires_data" in item.keywords:
            item.add_marker(pytest.mark.requires_data)

        # Mark slow tests
        if "slow" in item.keywords or "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)
