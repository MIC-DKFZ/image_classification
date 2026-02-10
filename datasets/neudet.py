

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule


class NEUDETData(Dataset):
    def __init__(
        self,
        root,
        split,
        transform=None,
        split_file="splits.json",
        labels_file="labels.json",
        allowed_exts=(".jpg", ".jpeg", ".png", ".bmp"),
        strict=True,
    ):
        """
        NEU-DET (NEU Surface Defect Database) loader.

        Folder layout:
            root/
              train/
                crazing_1.jpg
                inclusion_10.jpg
                ...
              validation/
                patches_5.jpg
                ...
              labels.json       {"train/crazing_1.jpg": 0, ...}
              splits.json       {"train":[...], "val":[...], "test":[...]}

        Args:
            split: "train" | "val" | "test"
            transform: optional callable
            strict: if True, raise on missing labels/files; else skip them.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
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

            # img_id format: "train/crazing_1.jpg" or "validation/patches_5.jpg"
            path = self.root / img_id
            if not path.exists() or not path.is_file():
                if self.strict:
                    raise FileNotFoundError(f"Missing image for id '{img_id}' at {path}")
                continue

            kept_files.append(path)
            kept_labels.append(label_map[img_id])

        self.img_paths = kept_files
        self.labels = np.asarray(kept_labels, dtype=np.int64)

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


class NEUDETDataModule(BaseDataModule):
    def __init__(self, **params):
        super(NEUDETDataModule, self).__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = NEUDETData(
            self.data_path,
            split="train",
            transform=self.train_transforms,
        )
        self.val_dataset = NEUDETData(
            self.data_path,
            split="val",
            transform=self.test_transforms,
        )


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    from augmentation.policies.neudet import TrainTransform, TestTransform

    # Get DATA_ROOT from environment or use default
    data_root = os.environ.get("DATA_ROOT", "./data")

    print("="*80)
    print("Testing NEUDET Dataset")
    print(f"Using DATA_ROOT: {data_root}")
    print("="*80)

    # Get augmentation transforms (instantiate the classes)
    train_aug = TrainTransform()()
    val_aug = TestTransform()()

    # Test train set with augmentations
    print("\n[Train Set with Augmentations]")
    train_ds = NEUDETData(root=f"{data_root}/NEUDET", split="train", transform=train_aug)
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
    val_ds = NEUDETData(root=f"{data_root}/NEUDET", split="val", transform=val_aug)
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
    print("✓ NEUDET Dataset test completed successfully!")
    print("="*80)
