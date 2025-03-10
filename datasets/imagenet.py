from torchvision.datasets import ImageNet
from torch.utils.data import Subset, Dataset
import numpy as np
from .base_datamodule import BaseDataModule



class SubsetImageNet(Dataset):
    def __init__(self, dataset, indices):
        """
        Wraps a subset of ImageNet as an ImageNet-like dataset.
        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        """
        Fetch item using subset indices.
        """
        return self.dataset[self.indices[index]]

    def __len__(self):
        """
        Returns the length of the subset.
        """
        return len(self.indices)

    def classes(self):
        """
        Keep the original dataset's classes.
        """
        return self.dataset.classes

    def class_to_idx(self):
        """
        Keep the original dataset's class-to-index mapping.
        """
        return self.dataset.class_to_idx


class ImagenetDataModule(BaseDataModule):
    def __init__(self, **params):
        super(ImagenetDataModule, self).__init__(**params)
        self.data_fraction = params['data_fraction']

    def setup(self, stage: str = None):
        if "albumentations" in str(self.train_transforms.__class__):
            raise NotImplementedError
        else:
            full_train_dataset = ImageNet(self.data_path, split="train", transform=self.train_transforms)
            num_samples = int(len(full_train_dataset) * self.data_fraction)
            indices = np.random.choice(len(full_train_dataset), num_samples, replace=False)
            self.train_dataset = SubsetImageNet(full_train_dataset, indices)

        if "albumentations" in str(self.test_transforms.__class__):
            raise NotImplementedError
        else:
            self.val_dataset = ImageNet(self.data_path, split="val", transform=self.test_transforms)

