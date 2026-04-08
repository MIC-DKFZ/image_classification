from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class PrecomputedFeaturesDataset(Dataset):
    """HDF5-backed dataset for precomputed feature tensors.

    Expected file structure:
    - dataset `features`: shape `(N, D)`
    - dataset `labels`: shape `(N,)`
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Precomputed feature file not found: {self.file_path}")

        with h5py.File(self.file_path, "r") as f:
            if "features" not in f or "labels" not in f:
                raise KeyError(
                    f"{self.file_path} must contain HDF5 datasets 'features' and 'labels'."
                )
            self._length = int(f["features"].shape[0])
            self.feature_dim = int(f["features"].shape[1])
            self.labels = f["labels"][:].astype("int64")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        with h5py.File(self.file_path, "r") as f:
            features = torch.from_numpy(f["features"][index]).float()
            label = torch.tensor(self.labels[index], dtype=torch.long)
        return features, label
