import random
import functools
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, Subset
from typing import Optional


class BaseDataModule(LightningDataModule):
    def __init_subclass__(cls, **kwargs):
        """Wrapping the setup method of subclasses, ensuring that the dataset fraction is
        applied after the dataset-specific setup.
        """
        super().__init_subclass__(**kwargs)
        original_setup = cls.setup

        @functools.wraps(original_setup)
        def wrapped_setup(self, *args, **kwargs):
            original_setup(self, *args, **kwargs)
            self._maybe_apply_fraction()

        cls.setup = wrapped_setup

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
        data_fraction: Optional[float] = None,
        stratified: bool = True,
        *args,
        **kwargs
    ):
        super(BaseDataModule, self).__init__()

        self.data_path = Path(data_root_dir)
        self.batch_size = batch_size
        self.train_transforms = train_transforms()
        self.test_transforms = test_transforms()
        self.random_batches = random_batches
        self.num_workers = num_workers
        self.prepare_data_per_node = prepare_data_per_node
        self.fold = fold
        self.data_fraction = data_fraction
        self.stratified = stratified
        self._fraction_applied = False

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        pass
    
    def _get_targets(self, dataset):
        """Retrieve the labels from a dataset. This method should be overridden by
        subclasses if needed.
        """
        if isinstance(dataset, Subset):
            base_targets = self._get_targets(dataset.dataset)
            indices = np.array(dataset.indices)
            return np.array(base_targets)[indices]
        if hasattr(dataset, "targets"):
            return dataset.targets
        if hasattr(dataset, "labels"):
            return dataset.labels
        raise AttributeError(
            f"{dataset.__class__.__name__} does not expose targets or labels"
        )
    
    def _apply_fraction(self, dataset, fraction: float, stratify: bool):
        """Apply a data fraction with optional stratification."""
        if stratify:
            targets = np.array(self._get_targets(dataset))
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=fraction, random_state=42
            )
            idx, _ = next(splitter.split(np.zeros(len(targets)), targets))
        else:
            idx = np.random.choice(
                len(dataset), int(len(dataset) * fraction), replace=False
            )

        return torch.utils.data.Subset(dataset, idx)

    def _maybe_apply_fraction(self):
        if self._fraction_applied:
            return
        if not hasattr(self, "train_dataset"):
            return
        if self.data_fraction >= 1.0 or self.data_fraction is None:
            self._fraction_applied = True
            return
        self.train_dataset = self._apply_fraction(
            self.train_dataset, self.data_fraction, self.stratified
        )
        self._fraction_applied = True

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
