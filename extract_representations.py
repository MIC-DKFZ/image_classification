from pathlib import Path
import os

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from parsing_utils import make_omegaconf_resolvers


import h5py
import numpy as np


@hydra.main(version_base=None, config_path="./cli_configs", config_name="train")
def extract_features_hdf5(cfg):
    # delete automatically created hydra logger
    try:
        Path(
            "./main.log"
        ).unlink()
    except:
        pass
    
    print(OmegaConf.to_yaml(cfg))
    
    assert cfg.data.module.data_fraction == 1
    
    # shortcut for setting no Augmentations via num_cycles=0
    if cfg.num_cycles == 0:
        cfg.data.module.train_transforms = cfg.data.module.test_transforms

    # instantiate the model using this config
    model = instantiate(cfg.model)
    model.eval()
    model.to("cuda")
    # instantiate the dataset from the config
    datamodule = instantiate(cfg.data).module
    datamodule.setup()
    dataset = datamodule.val_dataset
    dataloader = datamodule.val_dataloader()

    # Infer feature size
    feature_dim = model.extract_features(
        torch.randn(1, *datamodule.val_dataset[0][0].shape).to("cuda")
    ).shape[-1]
    num_tokens = model.extract_features(
        torch.randn(1, *datamodule.val_dataset[0][0].shape).to("cuda")
    ).shape[1]

    for split in ("train", "test", "val"):
        if split == "train":
            dataset = datamodule.train_dataset
            dataloader = datamodule.train_dataloader()
            fname = f"{cfg.model.type.replace('/', '_').replace('.', '_')}_{cfg.data.module.name.lower()}_train_n{cfg.num_cycles}.h5"
            num_cycles = max(cfg.num_cycles, 1)
        elif split == "test":
            try:
                dataset = datamodule.test_dataset
                dataloader = datamodule.test_dataloader()
                fname = f"{cfg.model.type.replace('/', '_').replace('.', '_')}_{cfg.data.module.name.lower()}_test.h5"
                num_cycles = 1
            except AttributeError:
                print(f"No test dataset available for {datamodule.__class__.__name__}")
                continue
        elif split == "val":
            dataset = datamodule.val_dataset
            dataloader = datamodule.val_dataloader()
            fname = f"{cfg.model.type.replace('/', '_').replace('.', '_')}_{cfg.data.module.name.lower()}_val.h5"
            num_cycles = 1
        
        out_dir = Path(cfg.data.feature_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / fname
        
        with h5py.File(out_file, "w") as f:
            num_samples = len(dataset)
            if split == "train":
                num_samples = int(len(dataset) * num_cycles)
            print(f"{num_samples = }")

            dset_features = f.create_dataset(
                "features", shape=(num_samples, num_tokens, feature_dim), dtype="float32"
            )
            dset_labels = f.create_dataset("labels", shape=(num_samples,), dtype="int64")

            index = 0
            for _ in tqdm(range(num_cycles), desc="Cycles"):
                for batch in tqdm(dataloader, desc=f"{split.upper()} Batches"):
                    x, y = batch
                    x = x.to("cuda")
                    # Shape: (batch_size, num_tokens, feature_dim)
                    features = model.extract_features(x).detach().cpu().numpy()
                    batch_size = len(y)
                    dset_features[index : index + batch_size] = features
                    dset_labels[index : index + batch_size] = y.numpy()
                    index += batch_size


if __name__ == "__main__":
    make_omegaconf_resolvers()
    extract_features_hdf5()
