from pathlib import Path
from typing import Literal, Dict, Optional
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from .base_datamodule import BaseDataModule


class ELPVDataset(Dataset):
    """
    Binary ELPV dataset (no utils.load_dataset, no id2path.json).
    Label rule: 1 if defect_proba > 0 else 0.

    Requires in `root`:
      - splits_final.json   : list[fold] -> {"train":[ids], "val":[ids], "test":[ids]}
      - labels_probs.json   : { "<id>": <float> }
      - data/labels.csv     : columns [path, probability, type] (used only to get image paths)

    Parameters
    ----------
    root : str
    split : {"train","val","test"}
    fold : int
    transform : callable | None
        Called as transform(image=img)["image"], with img as (3,H,W) uint8 tensor.
    csv_path : str | None
        Custom path to labels.csv; defaults to <root>/data/labels.csv
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        fold: int,
        transform=None,
        csv_path: Optional[str] = None,
    ):
        assert split in ("train", "val", "test")
        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.transform = transform
        # --- splits ---
        splits_file = self.root / "splits_final.json"

        if not splits_file.exists():
            raise FileNotFoundError(f"Missing {splits_file}")
        with open(splits_file, "r") as f:
            folds = json.load(f)
        if not (0 <= fold < len(folds)):
            raise ValueError(f"fold must be in [0, {len(folds)-1}], got {fold}")
        self.ids = [int(i) for i in folds[fold][split]]

        # --- probabilities -> binary labels (1 if > 0 else 0) ---
        probs_file = self.root / "labels_probs.json"
        if not probs_file.exists():
            raise FileNotFoundError(f"Missing {probs_file}")
        with open(probs_file, "r") as f:
            probs_map: Dict[str, float] = json.load(f)

        max_id = max(self.ids + [int(k) for k in probs_map.keys()])
        self.probs = np.zeros(max_id + 1, dtype=np.float64)
        for k, v in probs_map.items():
            self.probs[int(k)] = float(v)
        self.labels = (self.probs > 0.0).astype(np.int64)

        # --- read image paths from labels.csv (no id2path.json needed) ---
        csv_path = csv_path or str(self.root / "data" / "labels.csv")
        # NOTE: dtype widths may need bumping if paths/types are longer.
        data = np.genfromtxt(
            csv_path,
            dtype=[("|S200"), ("<f8"), ("|S32")],
            names=["path", "probability", "type"],
        )
        # Ensure we always have a 1D array even for small files
        if data.shape == ():
            data = np.array([data], dtype=data.dtype)

        rel_paths = np.char.decode(data["path"]).tolist()
        # Build an index->path list; IDs are assumed to align with CSV row order.
        # If your splits/labels were created from this same CSV order, this matches perfectly.
        self.index_to_path = rel_paths  # list length == total #rows in CSV

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        sid = int(self.ids[index])

        # Load image on-demand using CSV path
        try:
            rel_path = self.index_to_path[sid]
        except IndexError:
            raise IndexError(
                f"ID {sid} out of range for labels.csv paths "
                f"(len={len(self.index_to_path)}). Ensure splits/labels were generated from the same CSV."
            )
        p = Path(rel_path)
        if not p.is_absolute():
            p = self.root / 'data' / rel_path
        img = Image.open(p).convert("RGB")
        label = int(self.labels[sid])

        if self.transform:
            x = self.transform(img)

        return x, label



class ELPVDataModule(BaseDataModule):
    def __init__(self, **params):
        super().__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = ELPVDataset(
            self.data_path, split="train", fold=self.fold, transform=self.train_transforms
        )
        self.val_dataset = ELPVDataset(
            self.data_path, split="val", fold=self.fold, transform=self.test_transforms
        )


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_ds = ELPVDataset(root="/home/d246a/Documents/data/elpv/", split="train", fold=0)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

    imgs, labels = next(iter(train_loader))
    print("Batch imgs:", imgs.shape, imgs.dtype, imgs.min().item(), imgs.max().item())
    print("Batch labels:", labels.shape, labels.dtype)
    print("Unique labels in batch:", torch.unique(labels, return_counts=True))