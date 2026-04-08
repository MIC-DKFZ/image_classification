from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedFeaturesDataset(Dataset):
    """HDF5-backed dataset for precomputed feature tensors.

    Supported file structures:
    - instance features:
      - `features`: shape `(N, D)`
      - `labels`: shape `(N,)`
    - fixed-size bags:
      - `features`: shape `(B, N, D)`
      - `labels`: shape `(B,)`
    - variable-size bags:
      - `features`: shape `(M, D)`
      - `labels`: shape `(B,)`
      - either `bag_ptr`: shape `(B + 1,)` or `bag_lengths`: shape `(B,)`
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        feature_key: str = "features",
        label_key: str = "labels",
        bag_ptr_key: str = "bag_ptr",
        bag_lengths_key: str = "bag_lengths",
    ):
        self.file_path = Path(file_path)
        self.feature_key = feature_key
        self.label_key = label_key
        self.bag_ptr_key = bag_ptr_key
        self.bag_lengths_key = bag_lengths_key
        if not self.file_path.exists():
            raise FileNotFoundError(f"Precomputed feature file not found: {self.file_path}")

        with h5py.File(self.file_path, "r") as f:
            if self.feature_key not in f or self.label_key not in f:
                raise KeyError(
                    f"{self.file_path} must contain HDF5 datasets {self.feature_key!r} and {self.label_key!r}."
                )
            features = f[self.feature_key]
            labels = f[self.label_key]
            self.labels = labels[:].astype("int64")
            self._feature_shape = tuple(int(dim) for dim in features.shape)
            self.is_bag_dataset = False
            self.is_variable_bag_dataset = False

            if features.ndim == 2 and self.bag_ptr_key in f:
                bag_ptr = f[self.bag_ptr_key][:].astype("int64")
                if bag_ptr.ndim != 1 or bag_ptr.shape[0] != labels.shape[0] + 1:
                    raise ValueError(
                        f"{self.bag_ptr_key!r} in {self.file_path} must have shape (num_bags + 1,)."
                    )
                self.bag_ptr = bag_ptr
                self._length = int(labels.shape[0])
                self.feature_dim = int(features.shape[1])
                self.is_bag_dataset = True
                self.is_variable_bag_dataset = True
            elif features.ndim == 2 and self.bag_lengths_key in f:
                bag_lengths = f[self.bag_lengths_key][:].astype("int64")
                if bag_lengths.ndim != 1 or bag_lengths.shape[0] != labels.shape[0]:
                    raise ValueError(
                        f"{self.bag_lengths_key!r} in {self.file_path} must have shape (num_bags,)."
                    )
                self.bag_ptr = np.concatenate([[0], np.cumsum(bag_lengths, dtype=np.int64)])
                self._length = int(labels.shape[0])
                self.feature_dim = int(features.shape[1])
                self.is_bag_dataset = True
                self.is_variable_bag_dataset = True
            elif features.ndim == 3:
                self._length = int(features.shape[0])
                self.feature_dim = int(features.shape[2])
                self.is_bag_dataset = True
                self.is_variable_bag_dataset = False
            elif features.ndim == 2:
                self._length = int(features.shape[0])
                self.feature_dim = int(features.shape[1])
            else:
                raise ValueError(
                    f"Unsupported feature shape {tuple(features.shape)} in {self.file_path}."
                )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        with h5py.File(self.file_path, "r") as f:
            features_ds = f[self.feature_key]
            if self.is_variable_bag_dataset:
                start = int(self.bag_ptr[index])
                end = int(self.bag_ptr[index + 1])
                features = torch.from_numpy(features_ds[start:end]).float()
            else:
                features = torch.from_numpy(features_ds[index]).float()
            label = torch.tensor(self.labels[index], dtype=torch.long)
        return features, label
