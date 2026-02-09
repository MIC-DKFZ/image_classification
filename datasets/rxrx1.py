

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule


class RxRx1Data(Dataset):
    def __init__(
        self,
        root,
        split,
        transform=None,
        images_dir="images",
        split_file="splits.json",
        labels_file="labels.json",
        channel=1,  # Default to channel 1 (out of 6 channels)
        allowed_exts=(".png", ".jpg", ".jpeg"),
        strict=True,
    ):
        """
        RxRx1 dataset loader.

        RxRx1 images are stored in a hierarchical structure:
        images/experiment/Plate{plate}/well_s{site}.png
        Example: images/HEPG2-01/Plate1/B02_s1.png

        Site IDs in metadata: "HEPG2-01_1_B02_1" (experiment_plate_well_site)
        Image path: images/HEPG2-01/Plate1/B02_s1.png

        Args:
            split: "train" | "val" | "test"
            transform: optional callable
            channel: (unused - kept for API compatibility)
            strict: if True, raise on missing labels/files; else skip them.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_dir = self.root / images_dir
        self.channel = channel
        self.allowed_exts = tuple(e.lower() for e in allowed_exts)
        self.strict = strict

        split_path = self.root / split_file
        labels_path = self.root / labels_file

        # Load split ids
        with open(split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Split '{split}' not in {split_path}. Keys: {list(splits.keys())}")

        self.site_ids = [str(x) for x in splits[split]]

        # Load labels
        with open(labels_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        # Build final file list + labels
        kept_files = []
        kept_labels = []

        for site_id in self.site_ids:
            if site_id not in label_map:
                if self.strict:
                    raise KeyError(f"Missing label for site_id '{site_id}' in {labels_path}")
                continue

            # Parse site_id: "HEPG2-01_1_B02_1" -> experiment=HEPG2-01, plate=1, well=B02, site=1
            parts = site_id.split("_")
            if len(parts) < 4:
                if self.strict:
                    raise ValueError(f"Invalid site_id format: '{site_id}'")
                continue

            experiment = parts[0]
            plate = parts[1]
            well = parts[2]
            site = parts[3]

            # Construct path: images/HEPG2-01/Plate1/B02_s1.png
            img_name = f"{well}_s{site}.png"
            img_path = self.img_dir / experiment / f"Plate{plate}" / img_name

            if not img_path.exists() or not img_path.is_file():
                if self.strict:
                    raise FileNotFoundError(f"Missing image for site_id '{site_id}' at {img_path}")
                continue

            kept_files.append(img_path)
            kept_labels.append(label_map[site_id])

        self.img_paths = kept_files
        self.labels = np.asarray(kept_labels, dtype=np.int64)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        y = int(self.labels[idx])

        # RxRx1 images are grayscale, convert to RGB by replicating
        img = Image.open(img_path).convert("RGB")  # PIL handles L->RGB conversion

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


class RxRx1DataModule(BaseDataModule):
    def __init__(self, **params):
        super(RxRx1DataModule, self).__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = RxRx1Data(
            self.data_path,
            split="train",
            transform=self.train_transforms,
        )
        self.val_dataset = RxRx1Data(
            self.data_path,
            split="val",
            transform=self.test_transforms,
        )


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    from augmentation.policies.rxrx1 import TrainTransform, TestTransform

    # Get DATA_ROOT from environment or use default
    data_root = os.environ.get("DATA_ROOT", "/home/d246a/Documents/data/SynergyUnitDatasets")

    print("="*80)
    print("Testing RxRx1 Dataset")
    print(f"Using DATA_ROOT: {data_root}")
    print("="*80)

    # Get augmentation transforms (instantiate the classes)
    train_aug = TrainTransform()()
    val_aug = TestTransform()()

    # Test train set with augmentations
    print("\n[Train Set with Augmentations]")
    train_ds = RxRx1Data(root=f"{data_root}/RxRx1", split="train", transform=train_aug)
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
    val_ds = RxRx1Data(root=f"{data_root}/RxRx1", split="val", transform=val_aug)
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
    print("✓ RxRx1 Dataset test completed successfully!")
    print("="*80)
