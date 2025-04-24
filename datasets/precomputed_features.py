from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import h5py

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


FNAME_FORMAT_FEATURES = "{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5"


class DefaultDict(defaultdict):  # for partial string formatting
    def __missing__(self, key):
        return f"{{{key}}}"  # Keep the placeholder in the final string


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
        self.data_fraction = params["data_fraction"]
        self.name = params["name"]
        self.fname = FNAME_FORMAT_FEATURES.format_map(
            DefaultDict(str, {
                "model": params["model_type"].replace('/', '_').replace('.', '_'),
                "dataset": params["name"],
                "imgsize": params["imgsize"],
                "precision": params["precision"],
            })
        )

    def setup(self, stage = None):
        self.train_dataset = HDF5Dataset(self.data_path / self.fname.format(split="train"))
        if self.data_fraction != 1:
            num_samples = int(len(self.train_dataset) * self.data_fraction)
            indices = np.random.choice(len(self.train_dataset), num_samples, replace=False)
            self.train_dataset = Subset(self.train_dataset, indices)
        
        self.val_dataset = HDF5Dataset(self.data_path / self.fname.format(split="val"))
        
        if self.name == "ILSVRC_2012":
            self.test_dataset = self.val_dataset
        else:
            self.test_dataset = HDF5Dataset(self.data_path / self.fname.format(split="test"))


# -----------------------------------------------------------------------------------

FNAME_FORMAT_FEATURES_JOINT_AGG = "agg_joint_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5"

class HDF5DataModuleJointTokenAgg(BaseDataModule):
    def __init__(self, **params):
        super().__init__(**params)
        self.data_fraction = params["data_fraction"]
        self.name = params["name"]
        self.fname = FNAME_FORMAT_FEATURES_JOINT_AGG.format_map(
            DefaultDict(str, {
                "model": params["model_type"].replace('/', '_').replace('.', '_'),
                "dataset": params["name"],
                "imgsize": params["imgsize"],
                "precision": params["precision"],
            })
        )

    def setup(self, stage = None):
        self.train_dataset = HDF5Dataset(self.data_path / self.fname.format(split="train"))
        if self.data_fraction != 1:
            num_samples = int(len(self.train_dataset) * self.data_fraction)
            indices = np.random.choice(len(self.train_dataset), num_samples, replace=False)
            self.train_dataset = Subset(self.train_dataset, indices)
        
        self.val_dataset = HDF5Dataset(self.data_path / self.fname.format(split="val"))
        
        if self.name == "ILSVRC_2012":
            self.test_dataset = self.val_dataset
        else:
            self.test_dataset = HDF5Dataset(self.data_path / self.fname.format(split="test"))
