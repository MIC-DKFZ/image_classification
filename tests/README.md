## Test Suite

Comprehensive pytest-based testing for the classification_downstream project.

## Structure

```
tests/
├── __init__.py               # Package initialization
├── conftest.py              # Shared fixtures and configuration
├── test_datasets.py         # Dataset class tests
├── test_augmentations.py    # Augmentation policy tests
└── test_datamodules.py      # Dataset factory / dataloader tests
```

## Running Tests

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=datasets --cov=augmentation
```

### Run Specific Test Categories

```bash
# Run only unit tests (fast, no data required)
pytest -m unit

# Run only integration tests (requires actual data)
pytest -m "integration and requires_data"

# Run only augmentation tests
pytest -m augmentation

# Run only datamodule tests
pytest -m datamodule

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Test datasets only
pytest tests/test_datasets.py

# Test augmentations only
pytest tests/test_augmentations.py

# Test datamodules only
pytest tests/test_datamodules.py
```

### Run Tests for Specific Dataset

```bash
# Test only AID dataset
pytest -k "AID"

# Test only ZooScanNet
pytest -k "ZooScanNet"

# Test multiple datasets
pytest -k "AID or Flowers102"
```

## Test Markers

Tests are organized with markers for easy filtering:

| Marker | Description | Example |
|--------|-------------|---------|
| `unit` | Fast tests, no external dependencies | `pytest -m unit` |
| `integration` | Tests requiring actual data | `pytest -m integration` |
| `requires_data` | Tests requiring dataset files | `pytest -m requires_data` |
| `slow` | Slow-running tests | `pytest -m "not slow"` |
| `augmentation` | Augmentation policy tests | `pytest -m augmentation` |
| `datamodule` | Dataset factory tests | `pytest -m datamodule` |

## Test Categories

### Dataset Tests (`test_datasets.py`)

#### TestDatasetStructure
- ✓ `test_dataset_class_exists` - Verify dataset class can be imported
- ✓ `test_dataset_has_required_methods` - Check `__init__`, `__len__`, `__getitem__`

#### TestDatasetLoading
- ✓ `test_dataset_can_load` - Load dataset with actual data
- ✓ `test_dataset_getitem` - Test item retrieval and format
- ✓ `test_dataset_with_transforms` - Test with augmentation transforms

#### TestDatasetSplits
- ✓ `test_all_splits_exist` - Verify train/val/test splits exist
- ✓ `test_splits_no_overlap` - Ensure splits don't overlap

### Augmentation Tests (`test_augmentations.py`)

#### TestAugmentationPolicies
- ✓ `test_policy_module_exists` - Verify policy module can be imported
- ✓ `test_train_transforms_callable` - Check train transforms are callable
- ✓ `test_val_transforms_callable` - Check val transforms are callable

#### TestAugmentationApplication
- ✓ `test_train_transforms_work` - Apply train transforms to mock data
- ✓ `test_val_transforms_work` - Apply val transforms to mock data
- ✓ `test_transforms_normalize_values` - Verify ImageNet normalization
- ✓ `test_transforms_preserve_channels` - Ensure 3 channels maintained

#### TestAugmentationDifferences
- ✓ `test_train_and_val_produce_different_sizes` - Compare train vs val outputs
- ✓ `test_train_is_stochastic` - Verify train augmentations are random
- ✓ `test_val_is_deterministic` - Verify val transforms are deterministic

### Dataset Factory Tests (`test_datamodules.py`)

#### TestDatasetFactoryStructure
- ✓ `test_dataset_registered` - Verify dataset exists in factory registry
- ✓ `test_dataset_config_class_exists` - Verify matching DataConfig exists

#### TestDatasetFactoryIntegration
- ✓ `test_factory_builds_dataloaders` - Build train/val/test loaders from config
- ✓ `test_factory_train_batch_shape` - Validate one train batch
- ✓ `test_factory_val_batch_shape` - Validate one val batch

## Fixtures

Common fixtures available in all tests (defined in `conftest.py`):

### Environment Fixtures
- `setup_environment` - Sets up DATA_ROOT environment variable
- `data_root` - Returns Path to DATA_ROOT
- `temp_dir` - Creates temporary directory for tests

### Dataset Fixtures
- `dataset_name` - Parametrized fixture for all dataset names
- `dataset_config` - Configuration dict for each dataset

### Mock Data Fixtures
- `mock_image` - Mock RGB image tensor (3, 224, 224)
- `mock_batch` - Mock batch of images and labels
- `mock_normalized_image` - Mock normalized image

### Parametrization Fixtures
- `transform_type` - Parametrized "train" or "val"

## Example Usage

### Test a Specific Dataset

```python
# Run all tests for AID dataset
pytest -k "AID" -v

# Run only unit tests for AID
pytest -k "AID" -m unit -v

# Run only integration tests for AID
pytest -k "AID" -m integration -v
```

### Test Augmentations

```python
# Test all augmentation policies
pytest tests/test_augmentations.py -v

# Test specific policy
pytest tests/test_augmentations.py -k "aid" -v

# Test only normalization
pytest tests/test_augmentations.py -k "normalize" -v
```

### Skip Tests Requiring Data

```python
# Run only tests that don't need actual data
pytest -m "unit and not requires_data"

# Run all except integration tests
pytest -m "not integration"
```

## Dataset-Specific Notes

### PCam
PCam tests will be skipped if H5 files haven't been extracted:
```bash
# Extract H5 files first
python datasets/helpers/pcam_split.py --root $DATA_ROOT/PCam

# Then run tests
pytest -k "PCam"
```

### DiabeticRetinopathy
Uses custom split format, some split tests are skipped.

## Writing New Tests

### Add Test for New Dataset

1. Add dataset config to `conftest.py`:
```python
"YourDataset": {
    "module": "your_module",
    "class": "YourDatasetData",
    "config_class": "YourDatasetConfig",
    "policy": "your_policy",
    "num_classes": 10,
    "expected_shape": (3, 224, 224),
}
```

2. Tests will automatically run for your dataset (parametrized)

### Add Custom Test

```python
import pytest

@pytest.mark.unit
def test_custom_functionality(dataset_config):
    """Test description."""
    # Your test code
    assert True
```

## Continuous Testing

While developing:
```bash
# Watch mode (requires pytest-watch)
ptw

# Run on file changes
pytest --looponfail
```

## Debugging Failed Tests

```bash
# Show local variables on failure
pytest --showlocals

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Verbose output with full tracebacks
pytest -vv --tb=long
```

## Coverage Reports

```bash
# Generate coverage report
pytest --cov=datasets --cov=augmentation --cov-report=html

# View coverage
open htmlcov/index.html
```
