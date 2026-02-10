import json, sys, os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO

# Remove current directory (or anything containing your local 'datasets' folder)
sys_path_backup = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath('.')]
# Import (with different name than local folder)
import datasets as hf_datasets
# Restore sys.path
sys.path = sys_path_backup


class PanNukeHF(Dataset):
    """
    PanNuke (Hugging Face) Dataset

    Args:
        root (str | Path): local cache directory for HF datasets (will be created if missing)
        split (str): "train" or "val"
        fold (int): which fold to hold out as validation (1, 2, or 3)
                    train = the other two folds concatenated, val = this fold
        transform (callable, optional): either torchvision-style (PIL->Tensor) or
                    albumentations-style (expects numpy via `transform(image=...)["image"]`)
    """
    def __init__(self, root, split: str, fold: int, transform: Optional[callable] = None):
        super().__init__()
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        #assert fold in {1, 2, 3}, "fold must be 1, 2, or 3"
        print(fold)
        fold += 1
        cache_dir = str(Path(root))
        if split == "val":
            self.ds = hf_datasets.load_dataset("RationAI/PanNuke", split=f"fold{fold}", cache_dir=cache_dir)
        else:  # train = the other two folds
            others = [f"fold{i}" for i in {1, 2, 3} - {fold}]
            parts = [hf_datasets.load_dataset("RationAI/PanNuke", split=s, cache_dir=cache_dir) for s in others]
            self.ds = hf_datasets.concatenate_datasets(parts)

        self.transform = transform
        # Class names are available under the 'tissue' feature (19 classes)
        self.class_names = getattr(self.ds.features.get("tissue"), "names", None)

    def __len__(self):
        return len(self.ds)

    def _apply_transform(self, img):
        if self.transform is None:
            return F.to_tensor(img)  # PIL -> Tensor [0,1]
        # Try albumentations-style first
        try:
            out = self.transform(image=np.asarray(img))
            img = out["image"]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            return img
        except TypeError:
            # Fallback: torchvision-style callable
            return self.transform(img)

    def __getitem__(self, idx):
        ex = self.ds[int(idx)]
        img = ex["image"]          # PIL.Image (256x256)
        label = int(ex["tissue"])  # 0..18
        img = self._apply_transform(img)
        return img, label
    

class PanNukeHFDataModule(BaseDataModule):
    def __init__(self, **params):
        super(PanNukeHFDataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = PanNukeHF(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = PanNukeHF(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )

if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    # Define a simple transform (resize + tensor)
    tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Create train/val datasets for fold 1
    import os
    hf_cache = os.environ.get("HF_HOME", "./hf-cache")
    train_ds = PanNukeHF(root=hf_cache, split="train", fold=1, transform=tfm)
    val_ds   = PanNukeHF(root=hf_cache, split="val", fold=1, transform=tfm)

    # Wrap in DataLoader
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    # Print some info
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")