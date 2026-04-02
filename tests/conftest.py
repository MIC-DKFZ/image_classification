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
        os.environ["DATA_ROOT"] = "./data"
    yield


@pytest.fixture
def data_root():
    """Get the DATA_ROOT path."""
    return Path(os.environ.get("DATA_ROOT", "./data"))


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
            "dataset_key": "aid",
            "module": "aid",
            "class": "AIDData",
            "config_class": "AIDConfig",
            "policy": "dataset_specific.aid",
            "num_classes": 30,
            "expected_shape": (3, 224, 224),
        },
        "ZooScanNet": {
            "dataset_key": "zooscannet",
            "module": "zooscannet",
            "class": "ZooScanNetData",
            "config_class": "ZooScanNetConfig",
            "policy": "dataset_specific.zooscannet",
            "num_classes": 116,
            "expected_shape": (3, 224, 224),
        },
        "ChestXray14": {
            "dataset_key": "chestxray14",
            "module": "chestxray14",
            "class": "ChestXray14Data",
            "config_class": "ChestXRay14Config",
            "policy": "dataset_specific.chestxray14",
            "num_classes": 15,
            "expected_shape": (3, 224, 224),
        },
        "NEUDET": {
            "dataset_key": "neudet",
            "module": "neudet",
            "class": "NEUDETData",
            "config_class": "NeuDetConfig",
            "policy": "dataset_specific.neudet",
            "num_classes": 6,
            "expected_shape": (3, 224, 224),
        },
        "RxRx1": {
            "dataset_key": "rxrx1",
            "module": "rxrx1",
            "class": "RxRx1Data",
            "config_class": "RxRx1Config",
            "policy": "dataset_specific.rxrx1",
            "num_classes": 1139,
            "expected_shape": (3, 224, 224),
        },
        "Flowers102": {
            "dataset_key": "flowers102",
            "module": "flowers102",
            "class": "Flowers102Data",
            "config_class": "Flowers102Config",
            "policy": "dataset_specific.flowers102",
            "num_classes": 102,
            "expected_shape": (3, 224, 224),
        },
        "RESISC45": {
            "dataset_key": "resisc45",
            "module": "resisc45",
            "class": "RESISC45Data",
            "config_class": "RESISC45Config",
            "policy": "dataset_specific.resisc45",
            "num_classes": 45,
            "expected_shape": (3, 224, 224),
        },
        "PCam": {
            "dataset_key": "pcam",
            "module": "pcam",
            "class": "PCamData",
            "config_class": "PCamConfig",
            "policy": "dataset_specific.pcam",
            "num_classes": 2,
            "expected_shape": (3, 224, 224),
        },
        "DiabeticRetinopathy": {
            "dataset_key": "diabetic_retina",
            "module": "diabetic_retina",
            "class": "EyePACSData",
            "config_class": "DiabeticRetinaConfig",
            "policy": "dataset_specific.diabetic_retina",
            "num_classes": 5,
            "expected_shape": (3, 224, 224),
        },
        "FGVCAircraft": {
            "dataset_key": "fgvc_aircraft",
            "module": "fgvc_aircraft",
            "class": "FGVCAircraftData",
            "config_class": "FGVCAircraftConfig",
            "policy": "dataset_specific.fgvc_aircraft",
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
