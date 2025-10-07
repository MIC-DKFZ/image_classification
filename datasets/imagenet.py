from torchvision.datasets import ImageNet
from .base_datamodule import BaseDataModule


class ImagenetDataModule(BaseDataModule):
    def __init__(self, data_fraction: float = 1., stratified: bool = True, **params):
        super(ImagenetDataModule, self).__init__(**params)
        self.data_fraction = data_fraction
        self.stratified = stratified

    def setup(self, stage: str = None):
        if "albumentations" in str(self.train_transforms.__class__):
            raise NotImplementedError
        if "albumentations" in str(self.test_transforms.__class__):
            raise NotImplementedError
        
        full_train_dataset = ImageNet(
            self.data_path, split="train", transform=self.train_transforms
        )
        self.train_dataset = self._apply_fraction(
            full_train_dataset, self.data_fraction, self.stratified
        )
        self.val_dataset = ImageNet(
            self.data_path, split="val", transform=self.test_transforms
        )
