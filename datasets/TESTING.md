# Dataset Testing Guide

All dataset classes now include comprehensive test blocks that verify:
- ✓ File access and loading
- ✓ Augmentation pipelines (train & val transforms)
- ✓ Batch creation and data types
- ✓ Label distributions

## Quick Start

### Test All Datasets

```bash
# Set data root
export DATA_ROOT=/home/d246a/Documents/data/SynergyUnitDatasets

# Run all tests
python test_all_datasets.py
```

### Test Individual Datasets

Each dataset .py file can be run standalone:

```bash
export DATA_ROOT=/home/d246a/Documents/data/SynergyUnitDatasets

# Test AID dataset
python -m datasets.aid

# Test ZooScanNet dataset
python -m datasets.zooscannet

# Test ChestXray14 dataset
python -m datasets.chestxray14

# Test NEUDET dataset
python -m datasets.neudet

# Test RxRx1 dataset
python -m datasets.rxrx1

# Test Flowers102 dataset
python -m datasets.flowers102

# Test RESISC45 dataset
python -m datasets.resisc45

# Test PCam dataset
python -m datasets.pcam

# Test DiabeticRetinopathy dataset
python -m datasets.diabetic_retina

# Test FGVCAircraft dataset
python -m datasets.fgvc_aircraft
```

## What Each Test Does

For every dataset, the test:

1. **Loads augmentation transforms**
   - Gets train transforms (with augmentations)
   - Gets val transforms (minimal augmentations)

2. **Tests train set** (2 batches of 16 images each)
   - Applies training augmentations
   - Loads batches
   - Verifies image shapes, dtypes, value ranges
   - Checks label distribution

3. **Tests val set** (2 batches of 16 images each)
   - Applies validation augmentations
   - Loads batches
   - Verifies image shapes, dtypes, value ranges
   - Checks label distribution

## Expected Output Format

```
================================================================================
Testing AID Dataset
================================================================================

[Train Set with Augmentations]
Total train samples: 6000

Batch 1:
  Images: torch.Size([16, 3, 224, 224]), dtype=torch.float32, min=-2.118, max=2.640
  Labels: torch.Size([16]), dtype=torch.int64
  Unique labels: [0, 5, 7, 12, 15, 18, 22, 25]

Batch 2:
  Images: torch.Size([16, 3, 224, 224]), dtype=torch.float32, min=-2.118, max=2.640
  Labels: torch.Size([16]), dtype=torch.int64
  Unique labels: [1, 3, 8, 11, 14, 19, 21, 27]

[Val Set with Augmentations]
Total val samples: 2000

Batch 1:
  Images: torch.Size([16, 3, 224, 224]), dtype=torch.float32, min=-2.118, max=2.640
  Labels: torch.Size([16]), dtype=torch.int64
  Unique labels: [2, 4, 6, 9, 13, 16, 20, 24]

Batch 2:
  Images: torch.Size([16, 3, 224, 224]), dtype=torch.float32, min=-2.118, max=2.640
  Labels: torch.Size([16]), dtype=torch.int64
  Unique labels: [0, 5, 10, 15, 17, 23, 26, 29]

================================================================================
✓ AID Dataset test completed successfully!
================================================================================
```

## Interpreting Results

### ✓ Success Indicators

- **Image shape**: Should be `[batch_size, 3, height, width]`
- **Image dtype**: Should be `torch.float32`
- **Value range**: Typically `-2.x` to `2.x` after ImageNet normalization
- **Labels dtype**: Should be `torch.int64`
- **No errors**: File loading and transforms work correctly

### ⚠️ Common Issues

**FileNotFoundError**: Dataset not preprocessed or wrong path
```bash
# Check DATA_ROOT is correct
echo $DATA_ROOT

# Verify dataset exists
ls -la $DATA_ROOT/AID/
```

**Missing splits.json**: Dataset preprocessing not run
```bash
# Run preprocessing for the dataset
python datasets/helpers/aid_split.py --root $DATA_ROOT/AID
```

**Import Error (augmentation policy)**: Augmentation policy missing
```bash
# Check augmentation policies exist
ls -la augmentation/policies/
```

**Value range unexpected**: Check normalization is applied
- Should see negative values if ImageNet normalization is applied
- If values are in [0, 255], normalization may be missing

## Dataset-Specific Notes

### PCam
⚠️ **Requires H5 extraction before testing**
```bash
# Extract H5 files to PNG images first
python datasets/helpers/pcam_split.py --root $DATA_ROOT/PCam
```

### ZooScanNet
- Has 116 classes, so unique labels output is truncated to first 10
- Large dataset (797k images), may take longer to load

### RxRx1
- Has 1,139 classes (siRNA treatments)
- Unique labels output is truncated to first 10
- Multi-channel microscopy images

### DiabeticRetinopathy
- Uses existing splits from prior preprocessing
- 5 severity levels (0-4)

### ChestXray14
- Patient-level splits (no patient overlap)
- 15 disease labels
- Large dataset (112k images)

## Troubleshooting

### Dataset path issues
```python
# In Python, check if dataset exists
from pathlib import Path
import os

data_root = Path(os.environ.get("DATA_ROOT", "/path/to/data"))
print(f"AID exists: {(data_root / 'AID').exists()}")
print(f"splits.json exists: {(data_root / 'AID' / 'splits.json').exists()}")
```

### Missing dependencies
```bash
# Install required packages
pip install torch torchvision pillow numpy h5py tqdm scikit-learn
```

### Augmentation issues
```python
# Test augmentation directly
from augmentation.policies.aid import get_train_transforms
import torch

transforms = get_train_transforms()
dummy_img = torch.randint(0, 256, (3, 500, 500), dtype=torch.uint8)
result = transforms(**{"image": dummy_img})
print(result["image"].shape, result["image"].dtype)
```

## Performance Tips

- Use `num_workers=2` for testing (default in test blocks)
- Increase to `num_workers=4-8` for actual training
- Set `pin_memory=True` when using GPU
- Use smaller batch sizes if running out of memory

## Adding Tests to New Datasets

If you add a new dataset, include the test block:

```python
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from augmentation.policies.YOUR_POLICY import get_train_transforms, get_val_transforms

    print("="*80)
    print("Testing YourDataset Dataset")
    print("="*80)

    train_aug = get_train_transforms()
    val_aug = get_val_transforms()

    # Test train set
    print("\n[Train Set with Augmentations]")
    train_ds = YourDatasetData(root="$DATA_ROOT/YourDataset", split="train", transform=train_aug)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    print(f"Total train samples: {len(train_ds)}")
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        if batch_idx >= 2:
            break
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images: {imgs.shape}, dtype={imgs.dtype}, min={imgs.min().item():.3f}, max={imgs.max().item():.3f}")
        print(f"  Labels: {labels.shape}, dtype={labels.dtype}")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")

    # Test val set
    print("\n[Val Set with Augmentations]")
    val_ds = YourDatasetData(root="$DATA_ROOT/YourDataset", split="val", transform=val_aug)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    print(f"Total val samples: {len(val_ds)}")
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        if batch_idx >= 2:
            break
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images: {imgs.shape}, dtype={imgs.dtype}, min={imgs.min().item():.3f}, max={imgs.max().item():.3f}")
        print(f"  Labels: {labels.shape}, dtype={labels.dtype}")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")

    print("\n" + "="*80)
    print("✓ YourDataset Dataset test completed successfully!")
    print("="*80)
```
