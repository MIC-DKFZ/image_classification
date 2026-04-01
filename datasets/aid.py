

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset



class AIDData(Dataset):
    def __init__(
        self,
        root,
        split,
        transform=None,
        images_dir="images",
        split_file="splits.json",
        labels_file="labels.json",
        allowed_exts=(".jpeg", ".jpg", ".png", ".tif", ".tiff"),
        strict=True,
    ):
        """
        AID (Aerial Image Dataset) loader.

        Folder layout:
            root/
              images/
                Airport/
                  airport_1.jpg
                  airport_2.jpg
                  ...
                Beach/
                  ...
              labels.json       {"Airport/airport_1": 0, ...}
              splits.json       {"train":[...], "val":[...], "test":[...]}

        Args:
            split: "train" | "val" | "test"
            transform: optional callable that matches style:
                       transform(**{"image": <tensor>})["image"]
            strict: if True, raise on missing labels/files; else skip them.
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

        # Load split ids
        with open(split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Split '{split}' not in {split_path}. Keys: {list(splits.keys())}")

        self.img_files = [str(x) for x in splits[split]]

        # Load labels
        with open(labels_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        # Build final file list + labels
        kept_files = []
        kept_labels = []

        for img_id in self.img_files:
            if img_id not in label_map:
                if self.strict:
                    raise KeyError(f"Missing label for image id '{img_id}' in {labels_path}")
                continue

            # img_id format: "Airport/airport_1"
            path = self._resolve_image_path(img_id)
            if path is None:
                if self.strict:
                    raise FileNotFoundError(f"Missing image for id '{img_id}' under {self.img_dir}")
                continue

            kept_files.append(path)
            kept_labels.append(label_map[img_id])

        self.img_paths = kept_files
        self.labels = np.asarray(kept_labels, dtype=np.int64)

    def _resolve_image_path(self, img_id: str) -> Path | None:
        """
        Resolve an image ID to an existing file path.
        img_id format: "Airport/airport_1"
        """
        # Try with common extensions
        base_path = self.img_dir / img_id
        for ext in self.allowed_exts:
            p = Path(str(base_path) + ext)
            if p.exists() and p.is_file():
                return p

        # Last resort: glob with stem
        stem = os.path.splitext(img_id)[0]
        candidates = list(self.img_dir.glob(f"{stem}.*"))
        for cand in candidates:
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
    from augmentation.policies.aid import build_test_transform, build_train_transform

    # Get DATA_ROOT from environment or use default
    data_root = os.environ.get("DATA_ROOT", "./data")

    print("="*80)
    print("Testing AID Dataset")
    print(f"Using DATA_ROOT: {data_root}")
    print("="*80)

    # Build augmentation transforms
    train_aug = build_train_transform()
    val_aug = build_test_transform()

    # Test train set with augmentations
    print("\n[Train Set with Augmentations]")
    train_ds = AIDData(root=f"{data_root}/AID", split="train", transform=train_aug)
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
    val_ds = AIDData(root=f"{data_root}/AID", split="val", transform=val_aug)
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
    print("✓ AID Dataset test completed successfully!")
    print("="*80)
