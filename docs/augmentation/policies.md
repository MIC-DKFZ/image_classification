# Augmentation Policies

Augmentation policies for all 10 datasets following best practices for each domain.

## Available Policies

### Remote Sensing
- **[aid.py](aid.py)** - AID (600×600 → 224) - Rotation-invariant
- **[resisc45.py](resisc45.py)** - RESISC45 (256×256 → 224) - Rotation-invariant

### Medical Imaging
- **[chestxray14.py](chestxray14.py)** - ChestXray14 (1024×1024 → 224) - Minimal aug, no rotation
- **[diabetic_retina.py](diabetic_retina.py)** - Diabetic Retinopathy (variable → 224) - Rotation-invariant

### Microscopy
- **[zooscannet.py](zooscannet.py)** - ZooScanNet (variable 24-4911px → 224) - Rotation-invariant
- **[rxrx1.py](rxrx1.py)** - RxRx1 (256×256 → 224) - Rotation-invariant
- **[pcam.py](pcam.py)** - PCam (96×96 → 224 or native) - Rotation-invariant

### Industrial
- **[neudet.py](neudet.py)** - NEU-DET (200×200 → 224) - Simple upscale

### Fine-Grained
- **[flowers102.py](flowers102.py)** - Flowers-102 (~500×500 → 224) - Standard aug
- **[fgvc_aircraft.py](fgvc_aircraft.py)** - FGVC-Aircraft (variable → 224) - Standard aug

### General
- **[elpv.py](elpv.py)** - ELPV (300×300 → 224)
- **[imagenet.py](imagenet.py)** - ImageNet baseline
- **[cifar.py](cifar.py)** - CIFAR baseline

---

## Usage

```python
from augmentation.policies.zooscannet import build_test_transform, build_train_transform

# For training
train_transform = build_train_transform()
train_dataset = ZooScanNetData(root="...", split="train", transform=train_transform)

# For validation/test
test_transform = build_test_transform()
val_dataset = ZooScanNetData(root="...", split="val", transform=test_transform)
```

---

## Key Principles

### 1. **ImageNet Normalization (Standard)**
All policies use ImageNet statistics for pretrained model adaptation:
```python
MEAN_IMGNET = (0.485, 0.456, 0.406)
STD_IMGNET = (0.229, 0.224, 0.225)
```

### 2. **Domain-Specific Augmentations**

| Domain | Horizontal Flip | Vertical Flip | Rotation | ColorJitter |
|--------|----------------|---------------|----------|-------------|
| Remote Sensing | ✅ | ✅ | ✅ 180° | ✅ Strong |
| Medical (X-ray) | ✅ | ❌ | ❌ | ✅ Subtle |
| Medical (Retinal) | ✅ | ✅ | ✅ 180° | ✅ Subtle |
| Microscopy | ✅ | ✅ | ✅ 180° | ✅ Moderate |
| Industrial | ✅ | ✅ | ❌ | ✅ Moderate |
| Fine-Grained | ✅ | ❌ | ❌ | ✅ Strong |

### 3. **Resize Strategy**

**Small native size (≤256px):**
- Use `Resize(224)` directly
- Examples: NEU-DET, RxRx1, RESISC45, ZooScanNet

**Large native size (≥500px):**
- Use `RandomResizedCrop(224)` for training
- Use `Resize(256) → CenterCrop(224)` for test
- Examples: AID, Flowers-102, ChestXray14, DiabeticRetinopathy, FGVC-Aircraft

**Special case - PCam (96px):**
- Option 1: `Resize(224)` for standard ViT
- Option 2: Keep native 96×96 (use `TrainTransformNative`)

### 4. **Augmentation Strength**

**Minimal (Medical X-rays):**
- Only horizontal flip
- Subtle ColorJitter (0.1)
- Preserve anatomical orientation

**Moderate (Microscopy):**
- All flips + 180° rotation
- Moderate ColorJitter (0.15)
- Rotation-invariant

**Standard (Natural Images):**
- Horizontal flip only
- Strong ColorJitter (0.2)
- Object-centric

**Strong (Aerial/Remote Sensing):**
- All flips + 180° rotation
- Strong ColorJitter (0.2)
- Rotation-invariant

---

## Transform Order (Important!)

### Training:
```python
1. Resize / RandomResizedCrop  # ← Do BEFORE rotation!
2. RandomHorizontalFlip
3. RandomVerticalFlip
4. RandomRotation(180)
5. ColorJitter
6. ToTensor
7. Normalize(MEAN_IMGNET, STD_IMGNET)
```

### Testing:
```python
1. Resize(256)
2. CenterCrop(224)  # if needed
3. ToTensor
4. Normalize(MEAN_IMGNET, STD_IMGNET)
```

**Why resize before rotation?**
- Avoids creating large empty corners
- Better handling of variable-size inputs
- More efficient

---

## Notes

- **No preprocessing needed**: All transforms are applied on-the-fly
- **Consistent normalization**: Always use ImageNet stats for pretrained models
- **Dataset-aware**: Augmentations respect domain characteristics
- **Test-time consistency**: Always use deterministic transforms for eval

See [../datasets/AUGMENTATION_STRATEGY.md](../../datasets/AUGMENTATION_STRATEGY.md) for detailed rationale.
