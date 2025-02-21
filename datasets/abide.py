import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class AbideData(Dataset):
    def __init__(self, root, split, fold, transform=None):
        super().__init__()
        """
        ABIDE Dataset
        """
        self.img_dir = Path(root) / "nnUNetResEncUNetLPlans_3d_fullres"
        label_file = Path(root) / "labelsTr.json"
        split_file = Path(root) / "splits_final.json"

        with open(split_file) as f:
            self.img_files = json.load(f)[fold]["train" if split == "train" else "val"]

        with open(label_file) as f:
            labels = json.load(f)
        self.labels = [labels[i][1] for i in self.img_files]

        self.transform = transform

    def __getitem__(self, idx):

        img, _ = Blosc2IO.load(self.img_dir / (self.img_files[idx] + ".b2nd"), mode="r")

        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = torch.from_numpy(img[...])

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class AbideDataModule(BaseDataModule):
    def __init__(self, **params):
        super(AbideDataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = AbideData(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = AbideData(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )
