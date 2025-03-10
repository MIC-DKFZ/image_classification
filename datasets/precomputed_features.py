import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import h5py

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class HDF5Dataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.file = None  # Will be opened in `__getitem__` for multi-worker safety
        with h5py.File(h5_file, "r") as f:
            self.num_samples = f["labels"].shape[0]  

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.h5_file, "r", swmr=True)  # Open file once per worker

        feature = torch.from_numpy(self.file["features"][index])
        label = torch.tensor(self.file["labels"][index], dtype=torch.long)
        return feature, label


class HDF5DataModule(BaseDataModule):
    def __init__(self, **params):
        super().__init__(**params)
        self.name = params['name']
        self.data_fraction = params['data_fraction']
        self.num_cycles = params['num_cycles']
        self.model_type = params['model_type']

    def setup(self, stage = None):
        self.train_dataset = HDF5Dataset(
            self.data_path / f"{self.model_type.replace('/', '_').replace('.', '_')}_{self.name.lower()}_train_n{self.num_cycles}.h5"
        )
        if self.data_fraction != 1:
            num_samples = int(len(self.train_dataset) * self.data_fraction)
            indices = np.random.choice(len(self.train_dataset), num_samples, replace=False)
            self.train_dataset = Subset(self.train_dataset, indices)
        
        self.val_dataset = HDF5Dataset(
            self.data_path / f"{self.model_type.replace('/', '_').replace('.', '_')}_{self.name.lower()}_val.h5"
        )
        
        if self.name == "ilsvrc_2012":
            self.test_dataset = self.val_dataset
        else:
            self.test_dataset = HDF5Dataset(
                self.data_path / f"{self.model_type.replace('/', '_').replace('.', '_')}_{self.name.lower()}_test.h5"
            )


class BloscDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = np.load(data_dir / "labels.npy")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature, _ = Blosc2IO.load(str(self.data_dir / f"features_{index}.b2nd"))
        label = self.labels[index]
        return feature[0, ...], label.item()


class BloscDataModule(BaseDataModule):
    def __init__(self, **params):
        super().__init__(**params)
        self.name = params['name']
        self.data_fraction = params['data_fraction']
        self.num_cycles = params['num_cycles']
        self.model_type = params['model_type']

    def setup(self, stage = None):
        self.train_dataset = BloscDataset(
            self.data_path / f"{self.model_type.replace('/', '_').replace('.', '_')}_{self.name.lower()}_train"
        )
        if self.data_fraction != 1:
            num_samples = int(len(self.train_dataset) * self.data_fraction)
            indices = np.random.choice(len(self.train_dataset), num_samples, replace=False)
            self.train_dataset = Subset(self.train_dataset, indices)
        
        self.val_dataset = BloscDataset(
            self.data_path / f"{self.model_type.replace('/', '_').replace('.', '_')}_{self.name.lower()}_val"
        )
        
        if self.name == "ilsvrc_2012":
            self.test_dataset = self.val_dataset
        else:
            self.test_dataset = BloscDataset(
                self.data_path / f"{self.model_type.replace('/', '_').replace('.', '_')}_{self.name.lower()}_test"
            )
