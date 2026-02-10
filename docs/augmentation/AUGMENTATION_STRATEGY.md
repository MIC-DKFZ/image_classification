# Augmentation Strategy for Dataset Adaptation

## Key Principles

### 1. **No Preprocessing Needed**
- ✅ **Do transforms on-the-fly** during training
- ❌ **Don't preprocess** images beforehand
- Uses torchvision transforms with ImageNet normalization for pretrained models

### 2. **ImageNet Normalization (Standard)**
```python
MEAN_IMGNET = (0.485, 0.456, 0.406)
STD_IMGNET = (0.229, 0.224, 0.225)
```
Use this for adapting pretrained models (MAE, ViT, ResNet, etc.)

### 3. **Resize Strategy**
Based on original image sizes and dataset characteristics:

| Dataset | Original Size | Target Size | Strategy |
|---------|--------------|-------------|----------|
| AID | 600×600 | 224 | Resize or RandomResizedCrop |
| ZooScanNet | Variable (24-4911px) | 224 | Resize (center crop if needed) |
| ChestXray14 | 1024×1024 | 224 | RandomResizedCrop |
| NEU-DET | 200×200 | 224 | Resize (upscale) |
| RxRx1 | 256×256 | 224 | RandomResizedCrop or Resize |
| Flowers-102 | ~500×500 | 224 | RandomResizedCrop |
| RESISC45 | 256×256 | 224 | RandomResizedCrop or Resize |
| PCam | 96×96 | 96 or 224 | Resize (keep native or upscale) |
| DiabeticRetinopathy | 1880-3264px | 224 | RandomResizedCrop |
| FGVC-Aircraft | 416-740px | 224 | RandomResizedCrop |

---

## Dataset-Specific Augmentation Policies

### **Group 1: Remote Sensing & Aerial Images**
**Datasets**: AID, RESISC45
**Characteristics**: Rotation-invariant, scale-invariant

**Train Augmentations:**
- RandomResizedCrop(224, scale=(0.8, 1.0))
- RandomHorizontalFlip()
- RandomVerticalFlip()
- RandomRotation(180) # Aerial images can be in any orientation
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

**Test:**
- Resize(256)
- CenterCrop(224)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

---

### **Group 2: Medical Images - X-rays**
**Datasets**: ChestXray14
**Characteristics**: No rotation, preserve anatomical orientation

**Train Augmentations:**
- Resize(256)
- RandomCrop(224) or CenterCrop(224)
- RandomHorizontalFlip() # Chest X-rays can be flipped
- ColorJitter(brightness=0.1, contrast=0.1) # Subtle
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

**Test:**
- Resize(256)
- CenterCrop(224)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

---

### **Group 3: Medical Images - Retinal**
**Datasets**: DiabeticRetinopathy
**Characteristics**: Rotation-invariant (fundus images)

**Train Augmentations:**
- RandomResizedCrop(224, scale=(0.9, 1.0))
- RandomHorizontalFlip()
- RandomVerticalFlip()
- RandomRotation(180)
- ColorJitter(brightness=0.1, contrast=0.1)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

**Test:**
- Resize(256)
- CenterCrop(224)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

---

### **Group 4: Microscopy Images**
**Datasets**: RxRx1, PCam, ZooScanNet
**Characteristics**: Rotation-invariant, high variability

**Train Augmentations:**
- Resize(224) or RandomResizedCrop(224, scale=(0.8, 1.0))
- RandomHorizontalFlip()
- RandomVerticalFlip()
- RandomRotation(180) # Microscopy is rotation-invariant
- ColorJitter(brightness=0.15, contrast=0.15) # Moderate
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

**Test:**
- Resize(224) # For PCam: Resize(96→224) or keep at 96
- CenterCrop(224) if needed
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

**Special Note for PCam:**
- PCam native resolution is 96×96
- Option 1: Keep at 96×96 (no upscaling)
- Option 2: Resize to 224 (standard ViT input)
- Consider model's minimum input size

---

### **Group 5: Industrial Defects**
**Datasets**: NEU-DET
**Characteristics**: Texture-based, rotation may not be meaningful

**Train Augmentations:**
- Resize(224)
- RandomHorizontalFlip()
- RandomVerticalFlip()
- ColorJitter(brightness=0.2, contrast=0.2)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

**Test:**
- Resize(224)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

---

### **Group 6: Fine-Grained Recognition**
**Datasets**: Flowers-102, FGVC-Aircraft
**Characteristics**: Object-centric, standard orientation

**Train Augmentations:**
- RandomResizedCrop(224, scale=(0.8, 1.0))
- RandomHorizontalFlip()
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

**Test:**
- Resize(256)
- CenterCrop(224)
- ToTensor()
- Normalize(MEAN_IMGNET, STD_IMGNET)

---

## Implementation Notes

### 1. **Avoid Over-Augmentation**
- Medical images: Minimal augmentation to preserve diagnostic features
- Natural images: Standard augmentation is safe
- Microscopy: Rotation-invariant augmentations are essential

### 2. **Consistent Test Transforms**
- Always use deterministic transforms for validation/test
- Typical: Resize(256) → CenterCrop(224)

### 3. **Special Cases**
- **ZooScanNet**: After adaptive filtering, smallest images are 24px
  - Upscaling 24→224 is ~9x, acceptable for ViT
  - Consider RandomResizedCrop to avoid artifacts

- **PCam**: 96×96 native
  - Either keep at 96×96 or resize to 224
  - Check if your model supports 96×96 input

- **NEU-DET**: 200×200 native
  - Upscaling 200→224 is 1.12x, minimal quality loss

### 4. **Order of Operations**
```python
# Correct order for train:
1. Resize / RandomResizedCrop  # Do this BEFORE rotation!
2. RandomHorizontalFlip
3. RandomVerticalFlip
4. RandomRotation (if applicable)
5. ColorJitter
6. ToTensor
7. Normalize

# Correct order for test:
1. Resize
2. CenterCrop (if needed)
3. ToTensor
4. Normalize
```

### 5. **Why Resize Before Rotation?**
- Rotating after RandomResizedCrop is safer
- Avoids creating large empty corners that need filling
- Better for variable-size datasets

---

## Summary Table

| Dataset | Train Transform | Test Transform | Special Notes |
|---------|----------------|----------------|---------------|
| AID | RandomResizedCrop(224) + Flip + Rotate180 | Resize(256)→Crop(224) | Rotation-invariant |
| ZooScanNet | Resize(224) + Flip + Rotate180 | Resize(224) | Handle 24px minimum |
| ChestXray14 | Resize(256)→Crop(224) + HFlip | Resize(256)→Crop(224) | Minimal aug |
| NEU-DET | Resize(224) + Flip | Resize(224) | Simple upscale |
| RxRx1 | Resize(224) + Flip + Rotate180 | Resize(224) | Rotation-invariant |
| Flowers-102 | RandomResizedCrop(224) + HFlip | Resize(256)→Crop(224) | Standard aug |
| RESISC45 | RandomResizedCrop(224) + Flip + Rotate180 | Resize(256)→Crop(224) | Rotation-invariant |
| PCam | Resize(224) + Flip + Rotate180 | Resize(224) | Consider keeping 96×96 |
| DiabeticRetinopathy | RandomResizedCrop(224) + Flip + Rotate180 | Resize(256)→Crop(224) | Rotation-invariant |
| FGVC-Aircraft | RandomResizedCrop(224) + HFlip | Resize(256)→Crop(224) | Standard aug |

---

## Next Steps

1. ✅ Create augmentation policy files for each dataset
2. ✅ Follow existing pattern in `augmentation/policies/`
3. ✅ Test with a few batches to verify transforms work correctly
