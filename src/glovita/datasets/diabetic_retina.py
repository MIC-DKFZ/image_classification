

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .blosc2io import Blosc2IO

class EyePACSData(Dataset):
    def __init__(
        self,
        root,
        split,
        transform=None,
        images_dir="train",
        split_file="splits.json",
        labels_file="trainLabels.csv",
        image_col="image",
        label_col="level",
        allowed_exts=(".jpeg", ".jpg", ".png", ".tif", ".tiff"),
        strict=True,
    ):
        """
        EyePACS / Kaggle DR Dataset with minimal splits.json.

        Folder layout (example):
            root/
              train/              (images)
              trainLabels.csv
              splits.json         {"train":[...], "val":[...], "test":[...]}

        Args:
            split: "train" | "val" | "test"
            transform: optional callable that matches your MRNet style:
                       transform(**{"image": <tensor or array>})["image"]
                       (e.g., Albumentations wrapper). If None, returns torch tensor.
            strict: if True, raise on missing labels/files; else silently skip them.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_dir = self.root / images_dir
        self.allowed_exts = tuple(e.lower() for e in allowed_exts)
        self.strict = strict

        split_path = self.root / split_file
        labels_path = self.root / labels_file

        # ---- load split ids ----
        with open(split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Split '{split}' not in {split_path}. Keys: {list(splits.keys())}")

        # IDs may be stored as "10_left" or "10_left.jpeg"
        self.img_files = [str(x) for x in splits[split]]

        # ---- load labels ----
        df = pd.read_csv(labels_path)
        if image_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"{labels_path} must contain columns '{image_col}' and '{label_col}'")

        df = df[[image_col, label_col]].dropna().copy()
        df[image_col] = df[image_col].astype(str)
        df[label_col] = df[label_col].astype(int)

        # Kaggle EyePACS labels typically keyed by stem (no extension)
        label_map = dict(zip(df[image_col], df[label_col]))

        # ---- build final file list + labels (filter/validate) ----
        kept_files = []
        kept_labels = []

        for raw_id in self.img_files:
            stem = os.path.splitext(raw_id)[0]

            if stem not in label_map:
                if self.strict:
                    raise KeyError(f"Missing label for image id '{stem}' in {labels_path}")
                continue

            path = self._resolve_image_path(raw_id)
            if path is None:
                if self.strict:
                    raise FileNotFoundError(f"Missing image for id '{raw_id}' under {self.img_dir}")
                continue

            kept_files.append(path)                # Path
            kept_labels.append(label_map[stem])    # int

        self.img_paths = kept_files
        self.labels = np.asarray(kept_labels, dtype=np.int64)

    def _resolve_image_path(self, raw_id: str) -> Path | None:
        """
        Resolve an ID to an existing file path.
        - If ID already includes extension and exists -> use it
        - Else try allowed extensions appended to stem
        """
        p = self.img_dir / raw_id
        if p.exists() and p.is_file():
            return p

        stem = os.path.splitext(raw_id)[0]
        for ext in self.allowed_exts:
            p2 = self.img_dir / f"{stem}{ext}"
            if p2.exists() and p2.is_file():
                return p2

        # Last resort: any file with matching stem (handles weird/uppercase ext)
        for cand in self.img_dir.glob(f"{stem}.*"):
            if cand.is_file() and cand.suffix.lower() in self.allowed_exts:
                return cand

        return None

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        y = int(self.labels[idx])

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            # Transform expects PIL Image or numpy array
            img = self.transform(img)
        else:
            # No transform - convert to tensor manually
            img_np = np.ascontiguousarray(np.array(img), dtype=np.uint8)
            img = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float()

        return img, y


    def __len__(self):
        return len(self.img_paths)



if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    from glovita.augmentation.policies.dataset_specific.diabetic_retina import build_test_transform, build_train_transform

    # Get DATA_ROOT from environment or use default
    data_root = os.environ.get("DATA_ROOT", "./data")

    print("="*80)
    print("Testing DiabeticRetinopathy Dataset")
    print(f"Using DATA_ROOT: {data_root}")
    print("="*80)

    # Build augmentation transforms
    train_aug = build_train_transform()
    val_aug = build_test_transform()

    # Test train set with augmentations
    print("\n[Train Set with Augmentations]")
    train_ds = EyePACSData(root=f"{data_root}/DiabeticRetinopathy", split="train", transform=train_aug)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    print(f"Total train samples: {len(train_ds)}")
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        if batch_idx >= 2:  # Test 2 batches
            break
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images: {imgs.shape}, dtype={imgs.dtype}, min={imgs.min().item():.3f}, max={imgs.max().item():.3f}")
        print(f"  Labels: {labels.shape}, dtype={labels.dtype}")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")

    # Test val set with augmentations
    print("\n[Val Set with Augmentations]")
    val_ds = EyePACSData(root=f"{data_root}/DiabeticRetinopathy", split="val", transform=val_aug)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    print(f"Total val samples: {len(val_ds)}")
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        if batch_idx >= 2:  # Test 2 batches
            break
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images: {imgs.shape}, dtype={imgs.dtype}, min={imgs.min().item():.3f}, max={imgs.max().item():.3f}")
        print(f"  Labels: {labels.shape}, dtype={labels.dtype}")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")

    print("\n" + "="*80)
    print("✓ DiabeticRetinopathy Dataset test completed successfully!")
    print("="*80)
