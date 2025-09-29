# dataset_aid.py
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .base_datamodule import BaseDataModule

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class AIDData(Dataset):
    """
    AID (30-class scene classification)

    Expects:
      AID_ROOT/
        images/<class>/<file>.(jpg|png|...)
        splits_final.json                # [ {train:[], val:[], test:[]}, ... ]
        labels.json (optional)           # { "<class>/<stem>": <int> }
        class_map.json (optional)        # { "<class>": <int> }

    If labels.json is missing, labels are inferred from folder names using
    class_map.json (or alphabetical order if class_map.json missing).
    """

    def __init__(self, root: str, split: str, fold: int, transform=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.transform = transform
        self.img_dir = self.root / "images"

        # Load splits
        folds = json.loads((self.root / "splits_final.json").read_text())
        assert 0 <= fold < len(folds), f"fold {fold} out of range"
        self.ids = folds[fold]["train" if split == "train" else "val"]

        # Load or infer labels
        labels_path = self.root / "labels.json"
        if labels_path.exists():
            labels = json.loads(labels_path.read_text())
            self.labels = np.array([labels[i] for i in self.ids], dtype=np.int64)
            # If you want class names later, you can invert class_map.json
            self.class_map = json.loads((self.root/"class_map.json").read_text()) if (self.root/"class_map.json").exists() else None
        else:
            # Build class map (deterministic alphabetical) or load if provided
            if (self.root/"class_map.json").exists():
                cls2idx = json.loads((self.root/"class_map.json").read_text())
            else:
                class_names = sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])
                cls2idx = {c: i for i, c in enumerate(class_names)}
            self.class_map = cls2idx
            # Infer labels from the id's prefix (before '/')
            self.labels = np.array([cls2idx[i.split('/', 1)[0]] for i in self.ids], dtype=np.int64)

    def __len__(self): return len(self.ids)

    def _find_file(self, sid: str) -> Path:
        # sid is "<class>/<stem>"
        c, stem = sid.split("/", 1)
        for ext in IMG_EXTS:
            p = self.img_dir / c / f"{stem}{ext}"
            if p.exists():
                return p
        # Fallback: brute-force search
        for p in (self.img_dir / c).glob(f"{stem}.*"):
            if p.suffix.lower() in IMG_EXTS:
                return p
        raise FileNotFoundError(f"Image for id '{sid}' not found in {self.img_dir / c}")

    def __getitem__(self, idx):
        sid = self.ids[idx]
        path = self._find_file(sid)
        img = Image.open(path).convert("RGB")  # AID is RGB
        #x = torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()

        if self.transform:
            x = self.transform()(img)

        y = int(self.labels[idx])
        return x, y


class AIDDataModule(BaseDataModule):
    def __init__(self, **params):
        super().__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = AIDData(
            self.data_path, split="train", fold=self.fold, transform=self.train_transforms
        )
        self.val_dataset = AIDData(
            self.data_path, split="val", fold=self.fold, transform=self.test_transforms
        )
