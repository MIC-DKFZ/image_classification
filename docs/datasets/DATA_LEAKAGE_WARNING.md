# ⚠️ Data Leakage Warning

## Critical: Respecting Official Dataset Splits

**Problem**: Several datasets have official train/validation/test splits that MUST be preserved to prevent data leakage. Merging these splits and creating new random splits will cause test samples to leak into training data, invalidating all experimental results.

## Affected Datasets

### 1. RESISC45 (CRITICAL)
- **Official splits**: train/ validation/ test/ folders
- **MUST USE**: Exact official split boundaries
- **Why**: Test set is the standard benchmark - mixing it with training data makes results non-comparable with literature
- **Fixed script**: `datasets/helpers/resisc45_split.py`

### 2. Flowers102
- **Official splits**: train/ valid/ folders (test/ is unlabeled)
- **Strategy**: Use train as-is, split valid into val+test
- **Why**: Preserves official train/valid boundary
- **Fixed script**: `datasets/helpers/flowers102_split.py`

### 3. NEUDET
- **Official splits**: train/ validation/ folders
- **Strategy**: Use train as-is, split validation into val+test
- **Why**: Preserves official train/validation boundary
- **Fixed script**: `datasets/helpers/neudet_split.py`

## What Was Wrong

The original split scripts (now backed up as `*_BUGGY_BACKUP.py`) were:
1. Collecting ALL images from train/validation/test folders
2. Merging them into one pool
3. Creating NEW random 60/20/20 splits
4. ❌ This causes test samples to appear in training data!

Example of the problematic code:
```python
# WRONG - causes data leakage!
for split_name in ["train", "validation", "test"]:
    image_data.extend(collect_images(split_name))
# Then randomly re-split all images
new_splits = random_split(image_data, [0.6, 0.2, 0.2])
```

## Correct Approach

**Respect official boundaries:**
```python
# CORRECT - preserves official splits
train_images = collect_images("train")  # Official train
valid_images = collect_images("valid")   # Official valid
test_images = collect_images("test")     # Official test

# Use official splits as-is (or split valid if no test exists)
```

## How to Use Fixed Scripts

### RESISC45 (Uses official train/validation/test):
```bash
python datasets/helpers/resisc45_split.py --root /path/to/RESISC45
```

### Flowers102 (Splits valid into val+test):
```bash
python datasets/helpers/flowers102_split.py --root /path/to/Flowers102 \
    --test_from_valid_frac 0.5
```

### NEUDET (Splits validation into val+test):
```bash
python datasets/helpers/neudet_split.py --root /path/to/NEUDET_ROOT \
    --test_from_valid_frac 0.5
```

## Verification

Each fixed script verifies:
- ✓ No overlap between train/val/test
- ✓ Official boundaries preserved
- ✓ Clear logging of split sources

Output includes warnings like:
```
⚠️  CRITICAL: Official test split preserved - NO DATA LEAKAGE!
✓ Verified: No data leakage between official splits
```

## Impact

**If you used the buggy scripts**: Your test results may be artificially inflated because the model saw test samples during training. You MUST:
1. Re-generate splits using fixed scripts
2. Re-train all models
3. Re-evaluate on truly held-out test data

## General Rule

**Always check if a dataset has official splits before creating your own!**
- Check dataset documentation
- Look for train/test folders
- Search for official split files (e.g., train.txt, test.txt)
- When in doubt, preserve existing structure
