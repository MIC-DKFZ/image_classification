import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_root_dir,
        name,
        batch_size,
        train_transforms,
        test_transforms,
        random_batches,
        num_workers,
        prepare_data_per_node,
        fold,
        *args,
        **kwargs
    ):
        super(BaseDataModule, self).__init__()

        self.data_path = Path(data_root_dir)  # / name
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.random_batches = random_batches
        self.num_workers = num_workers
        self.prepare_data_per_node = prepare_data_per_node
        self.fold = fold

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        if not self.random_batches:
            trainloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
            )

        else:
            print("RandomSampler with replacement is used!")
            random_sampler = RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=len(self.train_dataset),
            )
            trainloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
                sampler=random_sampler,
            )

        return trainloader

    def val_dataloader(self):
        valloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        return valloader

    def test_dataloader(self):
        testloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        return testloader

    def predict_dataloader(self):
        predictloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        return predictloader


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
