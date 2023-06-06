from torchvision.datasets import CIFAR10, CIFAR100
from .base_datamodule import BaseDataModule


class Cifar10Albumentation(CIFAR10):

    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class Cifar100Albumentation(CIFAR100):

    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    

class CIFAR10DataModule(BaseDataModule):
    def __init__(self, **params):
        super(CIFAR10DataModule, self).__init__(**params)

    def setup(self, stage: str):

        if not self.albumentation:
            self.train_dataset = CIFAR10(self.root, train=True, transform=self.train_transforms, download=True)
            self.val_dataset = CIFAR10(self.root, train=False, transform=self.test_transforms, download=True)
        else:
            self.train_dataset = Cifar10Albumentation(self.root, train=True, transform=self.train_transforms, download=True)
            self.val_dataset = Cifar10Albumentation(self.root, train=False, transform=self.test_transforms, download=True)

    






























