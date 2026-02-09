# Dataset Structure Documentation

## Overview

All 10 datasets have been reorganized with clean folder names and consistent structure. All datasets are located in: `$DATA_ROOT/SynergyUnitDatasets/`

## Dataset Reorganization Summary

| Original Path | New Path | Changes Made |
|--------------|----------|--------------|
| `AID` | `AID` | ✓ No changes (already clean) |
| `ZooScanNet/ZooScanNet` | `ZooScanNet` | ✓ Flattened nested structure<br>✓ Renamed `imgs/` → `images/` |
| `2025_ChestXray14` | `ChestXray14` | ✓ Removed year prefix |
| `neu-surface-defect-database/NEU-DET` | `NEUDET` | ✓ Flattened nested structure |
| `rxrx1_v1.0` | `RxRx1` | ✓ Removed version suffix |
| `pytorch-challange-flower-dataset` | `Flowers102` | ✓ Clean name<br>✓ Renamed `dataset/` → `images/` |
| `resisc45_images` | `RESISC45` | ✓ Removed suffix |
| `pcamv1-20260120T124959Z-3-001` | `PCam` | ✓ Clean name |
| `diabetic-retinopathy-detection` | `DiabeticRetinopathy` | ✓ Clean name |
| `fgvc-aircraft/.../data` | `FGVCAircraft` | ✓ Flattened deeply nested structure |

## Final Dataset Structures

### 1. AID (Aerial Image Dataset)
```
AID/
├── images/
│   ├── Airport/
│   ├── Beach/
│   └── ... (30 classes)
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 30
- **Total images**: 10,000
- **Split**: 6,000 train / 2,000 val / 2,000 test

### 2. ZooScanNet
```
ZooScanNet/
├── images/              [RENAMED from imgs/]
│   ├── Copepoda/
│   ├── Medusae/
│   └── ... (116 classes after filtering)
├── splits.json
├── labels.json
├── class_map.json
└── filtering_stats.json
```
- **Classes**: 116 (filtered from 120, adaptive size filtering applied)
- **Total images**: 797,061
- **Split**: 478,236 train / 159,412 val / 159,413 test
- **Special**: Adaptive minimum image size filtering (24px-64px based on class population)

### 3. ChestXray14
```
ChestXray14/
├── images/
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 15 (disease labels)
- **Total images**: 112,120
- **Split**: 68,749 train / 22,125 val / 21,246 test
- **Special**: Patient-level stratified splitting (no patient overlap between splits)

### 4. NEUDET (Steel Defect Detection)
```
NEUDET/
├── train/
│   └── images/
│       ├── crazing/
│       ├── inclusion/
│       └── ... (6 classes)
├── validation/
│   └── images/
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 6 (defect types)
- **Total images**: 1,800
- **Split**: 1,080 train / 360 val / 360 test
- **Note**: Maintains original train/validation folder structure

### 5. RxRx1 (Cell Microscopy)
```
RxRx1/
├── images/
│   ├── HEPG2-01/
│   ├── HEPG2-02/
│   └── ... (51 cell plates)
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 1,139 (siRNA treatments)
- **Total images**: 115,656
- **Split**: 60,918 train / 20,306 val / 34,432 test

### 6. Flowers102
```
Flowers102/
├── images/              [RENAMED from dataset/]
│   ├── train/
│   ├── valid/
│   └── test/
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 102 (flower species)
- **Total images**: 7,370
- **Split**: 4,422 train / 1,474 val / 1,474 test

### 7. RESISC45 (Remote Sensing)
```
RESISC45/
├── train/
├── validation/
├── test/
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 45 (scene categories)
- **Total images**: 31,500
- **Split**: 18,900 train / 6,300 val / 6,300 test
- **Note**: Maintains original train/validation/test folder structure

### 8. PCam (Histopathology)
```
PCam/
└── pcamv1/
    ├── camelyonpatch_level_2_split_train_x.h5
    ├── camelyonpatch_level_2_split_train_y.h5
    └── ...
```
- **Classes**: 2 (tumor present/absent)
- **Native size**: 96×96 patches
- **Note**: H5 files need extraction (not yet done)

### 9. DiabeticRetinopathy
```
DiabeticRetinopathy/
├── train/
│   ├── 10_left.jpeg
│   ├── 10_right.jpeg
│   └── ... (~35k images)
├── splits.json
└── trainLabels.csv
```
- **Classes**: 5 (severity levels 0-4)
- **Note**: Uses existing splits from prior preprocessing

### 10. FGVCAircraft
```
FGVCAircraft/
├── images/
│   ├── 0034309.jpg
│   ├── 0034958.jpg
│   └── ... (10,000 images)
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 100 (aircraft variants)
- **Total images**: 10,000
- **Split**: 6,000 train / 2,000 val / 2,000 test

## File Format Standards

### splits.json
Standard format across all datasets:
```json
{
  "train": ["image_id1", "image_id2", ...],
  "val": ["image_id3", "image_id4", ...],
  "test": ["image_id5", "image_id6", ...]
}
```

### labels.json
Standard format across all datasets:
```json
{
  "image_path1": 0,
  "image_path2": 1,
  ...
}
```
- Keys: String image paths (relative to dataset root or images folder)
- Values: Integer class labels (0-indexed)

### class_map.json
Standard format (where applicable):
```json
{
  "class_name1": 0,
  "class_name2": 1,
  ...
}
```

## Dataset .py File Updates

All dataset classes have been updated to use the new clean folder names:

| Dataset File | Updated Path | Additional Changes |
|-------------|--------------|-------------------|
| `aid.py` | `$DATA_ROOT/AID` | None |
| `zooscannet.py` | `$DATA_ROOT/ZooScanNet` | `images_dir="imgs"` → `images_dir="images"` |
| `chestxray14.py` | `$DATA_ROOT/ChestXray14` | None |
| `neudet.py` | `$DATA_ROOT/NEUDET` | None |
| `rxrx1.py` | `$DATA_ROOT/RxRx1` | None |
| `flowers102.py` | `$DATA_ROOT/Flowers102` | `dataset_dir="dataset"` → `dataset_dir="images"` |
| `resisc45.py` | `$DATA_ROOT/RESISC45` | None |
| `pcam.py` | `$DATA_ROOT/PCam` | None |
| `diabetic_retina.py` | `$DATA_ROOT/DiabeticRetinopathy` | None |
| `fgvc_aircraft.py` | `$DATA_ROOT/FGVCAircraft` | None |

## Usage

Set the `$DATA_ROOT` environment variable to point to your SynergyUnitDatasets folder:

```bash
export DATA_ROOT=/home/d246a/Documents/data/SynergyUnitDatasets
```

Or in Python:
```python
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/SynergyUnitDatasets")

# Load a dataset
from datasets.aid import AIDDataModule
dm = AIDDataModule(data_path=f"{DATA_ROOT}/AID", ...)
```

## Notes

- All datasets use 60/20/20 train/val/test splits (where applicable)
- Stratified sampling ensures balanced class distribution across splits
- Patient-level splitting used for ChestXray14 to prevent data leakage
- Adaptive image size filtering used for ZooScanNet
- All unnecessary files backed up to `/home/d246a/Documents/data/backup_synergy/`
