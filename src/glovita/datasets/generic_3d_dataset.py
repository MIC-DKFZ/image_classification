import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from glovita.datasets.blosc2io import Blosc2IO


def load_split(split_path: Path, fold: str | None = None) -> dict:
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if isinstance(splits, list):
        return splits[int(fold) if fold is not None else 0]
    return splits


class Generic3DDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        transform=None,
        fold=None,
        subtask="multiclass",
        images_dir="images",
        split_file="splits.json",
        labels_file="labels.json",
    ):
        """
        Generic 3D volume dataset for preprocessed blosc2 files.

        Folder layout:
            root/
              images/
                case_001.b2nd       channel-first volume (C, X, Y, Z)
                ...
              dataset.json      {"num_classes": 3, "subtask": "multiclass"}
              labels.json       {"case_001": [0, 1, 0], ...}
              splits.json       {"train": [...], "val": [...], "test": [...]} or a list of these per fold

        Labels:
            multiclass: one-hot list (argmax is used) or integer class index
            multilabel: multi-hot list

        Args:
            split: "train" | "val" | "test"
            transform: optional callable applied to the volume tensor
            fold: fold index used when splits.json contains a list of folds
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.subtask = subtask
        self.img_dir = self.root / images_dir

        splits = load_split(self.root / split_file, fold)
        if split not in splits:
            raise ValueError(f"Split '{split}' not in {self.root / split_file}. Keys: {list(splits.keys())}")
        self.case_ids = [str(x) for x in splits[split]]

        with open(self.root / labels_file, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        if subtask == "multilabel":
            self.labels = np.asarray([label_map[case_id] for case_id in self.case_ids], dtype=np.int64)
        else:
            self.labels = np.asarray(
                [
                    int(np.argmax(label_map[case_id])) if isinstance(label_map[case_id], list) else int(label_map[case_id])
                    for case_id in self.case_ids
                ],
                dtype=np.int64,
            )

    def __getitem__(self, idx):
        img, _ = Blosc2IO.load(str(self.img_dir / f"{self.case_ids[idx]}.b2nd"), mode="r")
        img = torch.from_numpy(img[...])

        if self.transform:
            img = self.transform(img)

        if self.subtask == "multilabel":
            y = torch.from_numpy(self.labels[idx])
        else:
            y = int(self.labels[idx])
        return img, y

    def __len__(self):
        return len(self.case_ids)
