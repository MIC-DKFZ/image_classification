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

## Dataset Split Statistics

Quick reference table for all dataset splits:

| # | Dataset | Classes | Total Images | Train | Val | Test | Split Type |
|---|---------|---------|--------------|-------|-----|------|------------|
| 1 | AID | 30 | 10,000 | 6,000 | 2,000 | 2,000 | Random 60/20/20 |
| 2 | ZooScanNet | 116 | 797,061 | 478,236 | 159,412 | 159,413 | Random 60/20/20 (filtered) |
| 3 | ChestXray14 | 15 | 112,120 | 68,749 | 22,125 | 21,246 | Patient-level 60/20/20 |
| 4 | NEUDET | 6 | 1,800 | 1,440 | 180 | 180 | Official train + split valid |
| 5 | RxRx1 | 1,139 | 115,656 | 60,918 | 20,306 | 34,432 | Official train/test + split |
| 6 | Flowers102 | 102 | 7,370 | 6,552 | 490 | 328 | Official train + split valid |
| 7 | RESISC45 | 45 | 31,500 | 18,900 | 6,300 | 6,300 | Official train/val/test |
| 8 | PCam | 2 | 262,144 | 157,286 | 52,429 | 52,429 | Random 60/20/20 |
| 9 | DiabeticRetinopathy | 5 | 35,126 | 21,074 | 7,026 | 7,026 | Random 60/20/20 |
| 10 | FGVCAircraft | 100 | 10,000 | 6,000 | 2,000 | 2,000 | Random 60/20/20 |

**Notes:**
- **RESISC45**: Official splits MUST be preserved for benchmark comparability
- **Flowers102**: Non-stratified split (classes have 1-2 samples in valid set)
- **NEUDET**: Respects official train/validation boundary
- **ChestXray14**: Patient-level splitting prevents data leakage
- **ZooScanNet**: Adaptive size filtering applied (24px-64px based on class size)

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
- **Split**: 1,440 train / 180 val / 180 test
- **Note**: Uses official train folder, splits official validation into val+test (50/50)

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
│   └── test/           (unlabeled - not used)
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 102 (flower species)
- **Total images**: 7,370 (6,552 train + 818 valid, 819 test unlabeled)
- **Split**: 6,552 train / 490 val / 328 test
- **Note**: Uses official train folder, splits official valid into val (60%) + test (40%) with non-stratified split

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
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── splits.json
├── labels.json
└── class_map.json
```
- **Classes**: 2 (tumor present/absent)
- **Total images**: 262,144 (96×96 patches)
- **Split**: 157,286 train / 52,429 val / 52,429 test
- **Note**: Extracted from H5 files to PNG images

### 9. DiabeticRetinopathy
```
DiabeticRetinopathy/
├── train/
│   ├── 10_left.jpeg
│   ├── 10_right.jpeg
│   └── ...
├── splits.json
├── labels.json
├── class_map.json
└── trainLabels.csv
```
- **Classes**: 5 (severity levels 0-4)
- **Total images**: 35,126
- **Split**: 21,074 train / 7,026 val / 7,026 test

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
| `flowers102.py` | `$DATA_ROOT/Flowers102` | Removed `dataset_dir` parameter, uses `root/img_id` directly |
| `resisc45.py` | `$DATA_ROOT/RESISC45` | None |
| `pcam.py` | `$DATA_ROOT/PCam` | None |
| `diabetic_retina.py` | `$DATA_ROOT/DiabeticRetinopathy` | None |
| `fgvc_aircraft.py` | `$DATA_ROOT/FGVCAircraft` | None |

## Usage

Set the `$DATA_ROOT` environment variable to point to your SynergyUnitDatasets folder:

```bash
export DATA_ROOT=/path/to/your/SynergyUnitDatasets
```

Or in Python:
```python
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")

# Load a dataset
from datasets.aid import AIDDataModule
dm = AIDDataModule(data_path=f"{DATA_ROOT}/AID", ...)
```

## Notes

- Most datasets use 60/20/20 train/val/test splits with stratified sampling
- **RESISC45**: Uses official train/validation/test splits (18,900/6,300/6,300)
- **Flowers102**: Uses official train, splits official valid into val+test with non-stratified split (6,552/490/328)
- **NEUDET**: Uses official train, splits official validation into val+test (1,440/180/180)
- **ChestXray14**: Patient-level stratified splitting to prevent data leakage (68,749/22,125/21,246)
- **ZooScanNet**: Adaptive image size filtering applied (478,236/159,412/159,413)
- **RxRx1**: Uses existing train/test split, creates val from train (60,918/20,306/34,432)

