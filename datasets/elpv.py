import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from .base_datamodule import BaseDataModule


class ELPVData(Dataset):
    """
    ELPV (grayscale 300x300), binary or multi-class defect labels
    Expects:
      images/<id>.png
      labels.json -> { "<id>": <int> }
      splits_final.json
    """

    def __init__(self, root: str, split: str, fold: int, transform=None):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        with open(self.root / "splits_final.json") as f:
            folds = json.load(f)
        ids = folds[fold]["train" if split == "train" else "val"]
        with open(self.root / "labels.json") as f:
            labels = json.load(f)

        self.ids = [i for i in ids if i in labels]
        self.labels = np.array([labels[i] for i in self.ids], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        p = self.img_dir / f"{sid}.png"
        img = Image.open(p).convert("L")
        img = torch.from_numpy(np.array(img))  # (H, W)
        img = img.unsqueeze(0).repeat(3, 1, 1)  # grayscale -> 3ch

        if self.transform:
            img = self.transform(image=img)["image"]
        return img, int(self.labels[idx])


class ELPVDataModule(BaseDataModule):
    def __init__(self, **params):
        super().__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = ELPVData(
            self.data_path, split="train", fold=self.fold, transform=self.train_transforms
        )
        self.val_dataset = ELPVData(
            self.data_path, split="val", fold=self.fold, transform=self.test_transforms
        )
