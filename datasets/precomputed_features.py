from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import h5py

from .base_datamodule import BaseDataModule


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
    FNAME_FORMAT_FEATURES = "{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5"
    
    def __init__(self, data_fraction: float = 1., stratified: bool = True, **params):
        super().__init__(**params)
        self.data_fraction = data_fraction
        self.stratified = stratified
        self.name = params["name"]
        self.fname = self.FNAME_FORMAT_FEATURES.format_map(
            DefaultDict(str, {
                "model": params["model_type"].replace('/', '_').replace('.', '_'),
                "dataset": params["name"],
                "imgsize": params["imgsize"],
                "precision": params["precision"],
            })
        )
    
    def _get_targets(self, dataset: HDF5Dataset):
        with h5py.File(dataset.h5_file, "r") as f:
            labels = f["labels"][:]
        return labels

    def setup(self, stage = None):
        dset = HDF5Dataset(self.data_path / self.fname.format(split="train"))
        
        self.train_dataset = self._apply_fraction(
            dset, self.data_fraction, self.stratified
        )
        
        self.val_dataset = HDF5Dataset(self.data_path / self.fname.format(split="val"))
        
        if self.name == "ILSVRC_2012":
            self.test_dataset = self.val_dataset
        else:
            self.test_dataset = HDF5Dataset(self.data_path / self.fname.format(split="test"))


class HDF5DataModuleJointTokenAgg(HDF5DataModule):
    FNAME_FORMAT = "agg_joint_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5"
