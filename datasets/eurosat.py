from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import EuroSAT
from .base_datamodule import BaseDataModule
from torchvision.datasets.utils import download_url

def download_split_files_if_not_exists(eurosat_path: Path, split: str):
    split_file = eurosat_path / f"eurosat-{split}.txt"
    if not split_file.exists():
        download_url(
            f"https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/eurosat-{split}.txt",
            eurosat_path,
        )

class EuroSATDataModule(BaseDataModule):
    def __init__(self, data_fraction: float=1, **params):
        super(EuroSATDataModule, self).__init__(**params)
        self.data_fraction = data_fraction

    def setup(self, stage: str = None):
        if "albumentations" in str(self.train_transforms.__class__):
            raise NotImplementedError
        else:
            full_dataset = EuroSAT(self.data_path, download=True, transform=self.train_transforms)
            eurosat_path = Path(self.data_path) / "eurosat"
            for split in ["train", "val", "test"]:
                download_split_files_if_not_exists(eurosat_path, split)
                with open(eurosat_path / f"eurosat-{split}.txt", "r") as f:
                    split_files = f.readlines()
                split_files = [str(eurosat_path / "2750" / i.split("_")[0] / i.strip()) 
                                for i in split_files]
                indices = [i for i, s in enumerate(full_dataset.samples) if s[0] in split_files]
                subset = torch.utils.data.Subset(full_dataset, indices)
                setattr(self, f"{split}_dataset", subset)
        num_samples = int(len(self.train_dataset) * self.data_fraction)
        indices = np.random.choice(len(self.train_dataset), num_samples, replace=False)
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)

        if "albumentations" in str(self.test_transforms.__class__):
            raise NotImplementedError
        else:
            self.val_dataset.transform = self.test_transforms
            self.test_dataset.transform = self.test_transforms

