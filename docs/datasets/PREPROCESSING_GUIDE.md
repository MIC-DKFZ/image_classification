# Dataset Preprocessing Guide

This guide explains how to preprocess all 10 datasets using the provided scripts.

## Datasets Overview

All datasets will use **60/20/20 train/val/test splits** with stratified sampling.

**Note:** Replace `$DATA_ROOT` in the commands below with your actual dataset root directory path.

### Dataset List

1. **AID** - Aerial Image Dataset
2. **ZooScanNet** - Zooplankton classification
3. **ChestXray14** - Chest X-ray pathology
4. **NEU-DET** - Steel surface defects
5. **RxRx1** - Cell microscopy images
6. **Flowers-102** - Flower species
7. **RESISC45** - Remote sensing images
8. **PCam** - Histopathologic cancer detection
9. **DiabeticRetinopathy** - Already done
10. **FGVC-Aircraft** - Fine-grained aircraft classification

---

## Preprocessing Instructions

### 1. AID Dataset

**Location:** `$DATA_ROOT/AID`

```bash
# Create splits
python datasets/helpers/aid_split.py \
  --root $DATA_ROOT/AID \
  --labels_json labels.json \
  --out_json splits.json \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --seed 42
```

---

### 2. ZooScanNet Dataset ⚠️

**Location:** `$DATA_ROOT/ZooScanNet/ZooScanNet`

**Note**: ZooScanNet has extreme image size variation (4px to 4911px). Use the adaptive filtering script:

```bash
# Create splits with ADAPTIVE filtering based on class population
# Large classes: min 64px, Medium classes: min 48-56px, Small classes: min 32-40px
python datasets/helpers/zooscannet_split_adaptive.py \
  --root $DATA_ROOT/ZooScanNet/ZooScanNet \
  --imgs_dir imgs \
  --out_json splits.json \
  --out_labels labels.json \
  --absolute_min 24 \
  --min_samples 50 \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --seed 42
```

This will:
- Reject images smaller than 24px (absolute minimum to avoid extreme upscaling)
- Apply adaptive size thresholds based on class size
- Keep classes with at least 50 samples after filtering
- Generate detailed filtering statistics

---

### 3. ChestXray14 Dataset

**Location:** `$DATA_ROOT/2025_ChestXray14`

```bash
# Create splits with patient-level splitting (no patient leakage)
python datasets/helpers/chestxray14_split.py \
  --root $DATA_ROOT/2025_ChestXray14 \
  --csv_file Data_Entry_2017_v2020.csv \
  --out_json splits.json \
  --out_labels labels.json \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --seed 42
```

---

### 4. NEU-DET Dataset

**Location:** `$DATA_ROOT/neu-surface-defect-database`

```bash
# Create splits
python datasets/helpers/neudet_split.py \
  --root $DATA_ROOT/neu-surface-defect-database \
  --neudet_dir NEU-DET \
  --out_json splits.json \
  --out_labels labels.json \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --seed 42
```

---

### 5. RxRx1 Dataset

**Location:** `$DATA_ROOT/rxrx1_v1.0`

```bash
# Create splits (uses existing train/test split, creates val from train)
python datasets/helpers/rxrx1_split.py \
  --root $DATA_ROOT/rxrx1_v1.0 \
  --metadata_csv metadata.csv \
  --out_json splits.json \
  --out_labels labels.json \
  --val_frac 0.25 \
  --seed 42
```

---

### 6. Flowers-102 Dataset

**Location:** `$DATA_ROOT/pytorch-challange-flower-dataset`

```bash
# Create splits
python datasets/helpers/flowers102_split.py \
  --root $DATA_ROOT/pytorch-challange-flower-dataset \
  --dataset_dir dataset \
  --out_json splits.json \
  --out_labels labels.json \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --seed 42
```

---

### 7. RESISC45 Dataset

**Location:** `$DATA_ROOT/resisc45_images`

```bash
# Create splits
python datasets/helpers/resisc45_split.py \
  --root $DATA_ROOT/resisc45_images \
  --out_json splits.json \
  --out_labels labels.json \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --seed 42
```

---

### 8. PCam Dataset

**Location:** `$DATA_ROOT/pcamv1-20260120T124959Z-3-001`

**IMPORTANT:** First unzip the test files:
```bash
cd $DATA_ROOT/pcamv1-20260120T124959Z-3-001/pcamv1
gunzip camelyonpatch_level_2_split_test_x.h5.gz
gunzip camelyonpatch_level_2_split_test_y.h5.gz
```

Then extract and create splits:
```bash
# This will extract H5 files to PNG images and create splits
python datasets/helpers/pcam_split.py \
  --root $DATA_ROOT/pcamv1-20260120T124959Z-3-001 \
  --pcam_dir pcamv1 \
  --images_dir images \
  --out_json splits.json \
  --out_labels labels.json \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --seed 42
```

---

### 9. DiabeticRetinopathy Dataset

**Already done!** Located at `$DATA_ROOT/diabetic-retinopathy-detection`

---

### 10. FGVC-Aircraft Dataset

**Location:** `$DATA_ROOT/fgvc-aircraft`

```bash
# Create splits
python datasets/helpers/fgvc_aircraft_split.py \
  --root $DATA_ROOT/fgvc-aircraft \
  --aircraft_dir fgvc-aircraft-2013b/fgvc-aircraft-2013b/data \
  --out_json splits.json \
  --out_labels labels.json \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --seed 42
```

---

## Output Files

Each preprocessing script creates:
- `splits.json` - Train/val/test image IDs
- `labels.json` - Image ID to label mapping
- `class_map.json` - Class name to index mapping

---

## Using the Dataset Classes

Each dataset has a corresponding Python module in `datasets/`:

```python
from torch.utils.data import DataLoader
from datasets.aid import AIDData

# Load dataset
train_ds = AIDData(
    root="$DATA_ROOT/AID",  # Replace with your actual path
    split="train",
    transform=None  # Add your transforms here
)

# Create dataloader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
```

Available dataset classes:
- `AIDData`, `AIDDataModule`
- `ZooScanNetData`, `ZooScanNetDataModule`
- `ChestXray14Data`, `ChestXray14DataModule`
- `NEUDETData`, `NEUDETDataModule`
- `RxRx1Data`, `RxRx1DataModule`
- `Flowers102Data`, `Flowers102DataModule`
- `RESISC45Data`, `RESISC45DataModule`
- `PCamData`, `PCamDataModule`
- `EyePACSData`, `EyePACSDataModule` (DiabeticRetinopathy)
- `FGVCAircraftData`, `FGVCAircraftDataModule`

---

## Testing

Each dataset module has a `__main__` section for testing:

```bash
python -m datasets.aid
python -m datasets.flowers102
# etc.
```

---

## Notes

- All splits use stratified sampling to maintain class balance
- ChestXray14 uses patient-level splitting to avoid data leakage
- PCam requires H5 extraction (may take time and disk space)
- ZooScanNet filters out small images and rare classes
- All datasets follow the same interface for consistency
