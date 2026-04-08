# Testing Quick Start

Two testing approaches available:

## 1. Quick Testing (Simple)

**Use the `__main__` blocks for quick sanity checks:**

```bash
export DATA_ROOT=/path_to_data/SynergyUnitDatasets

# Test individual dataset
python -m datasets.aid
python -m datasets.zooscannet
python -m datasets.chestxray14
# ... etc

# Test all datasets
python test_all_datasets.py
```

**What it does:**
- Loads 2 batches × 16 images
- Tests train & val transforms
- Shows shapes, dtypes, value ranges
- Quick debugging tool

## 2. Proper Testing (Professional)

**Use pytest for comprehensive testing:**

```bash
# Install pytest (if not installed)
pip install pytest pytest-env

# Run tests
pytest                          # All tests
pytest -m unit                  # Fast unit tests only
pytest -m integration          # Integration tests (requires data)
pytest tests/test_datasets.py  # Dataset tests only
pytest -k "AID"                # Specific dataset
```

**What it does:**
- Parametrized testing (all datasets automatically tested)
- Proper test organization
- Coverage reports
- CI/CD ready

## Test Markers

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest -m unit` | Fast tests, no data needed |
| `pytest -m integration` | Tests with actual data |
| `pytest -m augmentation` | Augmentation tests only |
| `pytest -k "AID"` | Test specific dataset |
| `pytest -v` | Verbose output |
| `pytest -s` | Show print statements |

## Common Workflows

### Before Committing Code
```bash
# Run unit tests (fast)
pytest -m unit

# Run all tests if you have time
pytest
```

### Debugging a Dataset
```bash
# Quick check
python -m datasets.aid

# Comprehensive check
pytest -k "AID" -v

# With debugger
pytest -k "AID" --pdb
```

### Testing Augmentations
```bash
# Quick check
python -c "
from glovita.augmentation.policies.dataset_specific.aid import build_train_transform
import torch
t = build_train_transform()
img = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()
result = t(img)
print(result.shape, result.dtype)
"

# Comprehensive check
pytest tests/test_augmentations.py -k "aid"
```

## File Organization

```
classification_downstream/
├── datasets/                  # Dataset classes
│   ├── aid.py                # Has __main__ test block
│   └── ...
├── tests/                    # Pytest tests
│   ├── test_datasets.py      # Dataset tests
│   ├── test_augmentations.py # Augmentation tests
│   └── test_datamodules.py   # Dataset factory tests
├── test_all_datasets.py     # Convenience script
├── pytest.ini               # Pytest config
└── TESTING_QUICK_START.md   # This file
```

## When to Use Which

### Use `__main__` blocks when:
- ✓ Quick debugging
- ✓ Checking if dataset loads
- ✓ Visual inspection of outputs
- ✓ No pytest installed

### Use pytest when:
- ✓ Before committing code
- ✓ Testing all datasets
- ✓ CI/CD pipeline
- ✓ Coverage reports needed
- ✓ Systematic testing

## Examples

### Example 1: Quick Check Before Training
```bash
# Make sure dataset loads correctly
python -m datasets.aid

# Output:
# ================================================================================
# Testing AID Dataset
# ================================================================================
# [Train Set with Augmentations]
# Total train samples: 6000
# Batch 1:
#   Images: torch.Size([16, 3, 224, 224]), dtype=torch.float32, min=-2.118, max=2.640
# ...
```

### Example 2: Comprehensive Testing
```bash
# Test everything
pytest -v

# Output shows:
# tests/test_datasets.py::TestDatasetStructure::test_dataset_class_exists[AID] PASSED
# tests/test_datasets.py::TestDatasetStructure::test_dataset_class_exists[ZooScanNet] PASSED
# ...
```

### Example 3: Test After Adding New Dataset
```bash
# Add new dataset to conftest.py
# Then run:
pytest -k "YourNewDataset" -v
```

## Troubleshooting

### Tests Skip (Missing Data)
```bash
# Check dataset exists
ls -la $DATA_ROOT/AID/

# Check splits.json exists
ls -la $DATA_ROOT/AID/splits.json

# Run preprocessing if needed
python datasets/helpers/aid_split.py --root $DATA_ROOT/AID
```

### Import Errors
```bash
# Ensure you're in project root
cd /home/to/projects/classification_downstream

# Ensure packages installed
pip install torch torchvision pillow numpy h5py tqdm scikit-learn

# Run pytest
pytest
```

### PCam Tests Fail
```bash
# Extract H5 files first
python datasets/helpers/pcam_split.py --root $DATA_ROOT/PCam

# Then test
pytest -k "PCam"
```
