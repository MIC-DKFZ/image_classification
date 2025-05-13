from pathlib import Path
import json

from torch.utils.data import Dataset
import torch

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO
from models.mil_utils import PatchExtractor


class RSNA_SpineData(Dataset):
    def __init__(
        self,
        root,
        split,
        fold,
        patch_size,
        sliding_window_step_size,
        pad_value,
        random_patches,
        num_patches,
        transform=None,
    ):
        super().__init__()
        """
        RSNA_Spine Dataset
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

        # init patch extractor
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            step_size=sliding_window_step_size,
            padding_value=pad_value,
            random=random_patches,
            num_random_patches=num_patches,
            batch_size=1,  # needs to be 1 for augmentations
        )

    def __getitem__(self, idx):

        img, _ = Blosc2IO.load(self.img_dir / (self.img_files[idx] + ".b2nd"), mode="r")

        # yield all patches
        img = torch.from_numpy(img[...]).unsqueeze(0)

        self.patch_extractor.set_array(img)
        patches = []
        for i in self.patch_extractor:
            i = i.squeeze().unsqueeze(0)
            if self.transform:
                i = self.transform(**{"image": i})["image"]

            patches.append(i)

        patches = torch.stack(patches)

        return patches, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class RSNA_SpineDataModule(BaseDataModule):
    def __init__(
        self,
        patch_size,
        sliding_window_step_size,
        pad_value,
        random_patches,
        num_patches,
        **params
    ):
        super(RSNA_SpineDataModule, self).__init__(**params)

        self.patch_size = patch_size
        self.sliding_window_step_size = sliding_window_step_size
        self.pad_value = pad_value
        self.random_patches = random_patches
        self.num_patches = num_patches

    def setup(self, stage: str):

        self.train_dataset = RSNA_SpineData(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
            patch_size=self.patch_size,
            sliding_window_step_size=self.sliding_window_step_size,
            pad_value=self.pad_value,
            random_patches=self.random_patches,
            num_patches=self.num_patches,
        )
        self.val_dataset = RSNA_SpineData(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
            patch_size=self.patch_size,
            sliding_window_step_size=self.sliding_window_step_size,
            pad_value=self.pad_value,
            random_patches=False,  # always sample all possible patches for validation
            num_patches=self.num_patches,
        )
