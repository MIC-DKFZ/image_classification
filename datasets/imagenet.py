from torchvision.datasets import ImageNet

from .base_datamodule import BaseDataModule


class ImagenetDataModule(BaseDataModule):
    def __init__(self, **params):
        super(ImagenetDataModule, self).__init__(**params)

    def setup(self, stage: str):
        if "albumentations" in str(self.train_transforms.__class__):
            raise NotImplementedError
        else:
            self.train_dataset = ImageNet(self.data_path, split="train", transform=self.train_transforms)

        if "albumentations" in str(self.test_transforms.__class__):
            raise NotImplementedError
        else:
            self.val_dataset = ImageNet(self.data_path, split="val", transform=self.test_transforms)
