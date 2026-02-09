# Image Size Analysis Across Datasets

## Summary

Analysis of image size distributions across all 10 datasets to determine filtering strategies.

---

## Datasets with UNIFORM Image Sizes (No Filtering Needed)

### 1. AID
- **Size**: 600 × 600 (all images)
- **Action**: No size filtering needed
- **Status**: ✅ Ready

### 2. RESISC45
- **Size**: 256 × 256 (all images)
- **Action**: No size filtering needed
- **Status**: ✅ Ready

### 3. Flowers-102
- **Size**: ~500 × 500 (all images)
- **Action**: No size filtering needed
- **Status**: ✅ Ready

### 4. ChestXray14
- **Size**: 1024 × 1024 (all images)
- **Action**: No size filtering needed
- **Status**: ✅ Ready

### 5. RxRx1
- **Size**: 256 × 256 (all images)
- **Action**: No size filtering needed
- **Status**: ✅ Ready

### 6. NEU-DET
- **Size**: 200 × 200 (all images)
- **Action**: No size filtering needed
- **Status**: ✅ Ready

---

## Datasets with VARIABLE Image Sizes

### 7. ZooScanNet ⚠️ **REQUIRES ADAPTIVE FILTERING**
- **Size Range**: **4px to 4911px** (extreme variation!)
- **Issues**:
  - Many classes have very small images (4-25px minimum)
  - Some classes are rare (few samples)
  - Upscaling 4×5 images to 224×224 would be terrible quality
- **Strategy**: Adaptive filtering based on class population
  - Large classes (>500 samples): min_size = 64px
  - Medium classes (100-500): min_size = 48-56px
  - Small classes (50-100): min_size = 40px
  - Very small classes (20-50): min_size = 32px
  - **Absolute minimum**: 24px (reject below this)
- **Script**: `zooscannet_split_adaptive.py`
- **Status**: ⚠️ Needs adaptive filtering

### 8. Diabetic Retinopathy
- **Size Range**: ~1880px to ~3264px
- **Issues**: Moderate variation but all images are large
- **Strategy**: Simple minimum threshold (optional)
  - Could apply min_size = 1500px if desired
  - Or no filtering since all are reasonably large
- **Status**: ✅ All images sufficiently large

### 9. FGVC-Aircraft
- **Size Range**: ~416px to 740+px
- **Issues**: Moderate variation but all images reasonable
- **Strategy**: Optional simple threshold
  - Could apply min_size = 400px if desired
  - Or no filtering since all are reasonably large
- **Status**: ✅ All images sufficiently large

### 10. PCam
- **Size**: 96 × 96 (extracted from H5 files)
- **Issues**: All images are small but uniform
- **Strategy**: No filtering (dataset designed at this resolution)
- **Status**: ✅ Uniform resolution

---

## Filtering Recommendations

### Critical (Must Apply)
- **ZooScanNet**: Use `zooscannet_split_adaptive.py` with adaptive filtering

### Optional (Can Apply)
- **Diabetic Retinopathy**: Simple min_size = 1500px (optional)
- **FGVC-Aircraft**: Simple min_size = 400px (optional)

### Not Needed
- All other datasets have uniform or sufficiently large images

---

## ZooScanNet Detailed Statistics

Sample of classes showing size variation:

| Class | Count | Min | Max | Mean | Median |
|-------|-------|-----|-----|------|--------|
| detritus | 241,731 | 4 | 5237 | 49.1 | 34 |
| Phaeodaria | 54,036 | 10 | 3199 | 117.8 | 85 |
| Copepoda | 43,886 | 8 | 3467 | 137.6 | 113 |
| Evadne | 33,348 | 32 | 155 | 73.6 | 72 |
| Penilia | 23,945 | 35 | 139 | 74.4 | 73 |
| Eumalacostraca | 23,015 | 17 | 4242 | 169.6 | 124 |

**Key Observations:**
- "detritus" class has images as small as **4×5 pixels**!
- Many classes have minimum sizes below 32px
- Large variation even within single classes
- Adaptive filtering essential to balance quality vs. class representation

---

## Implementation Status

✅ **Completed:**
- Image size analysis script (`analyze_dataset_sizes.py`)
- Adaptive filtering for ZooScanNet (`zooscannet_split_adaptive.py`)

⏳ **Optional:**
- Simple size filtering for Diabetic Retinopathy (if desired)
- Simple size filtering for FGVC-Aircraft (if desired)
