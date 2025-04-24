from pathlib import Path

import h5py
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from parsing_utils import make_omegaconf_resolvers


# from datasets.precomputed_features import FNAME_FORMAT_FEATURES
FNAME_FORMAT_FEATURES = "agg_joint_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5"


def aggregate_features(x, method: str):
    if method == "cls_token":
        x = x[:, 0]
    elif method == "avg":
        x = x[:, 1:].mean(dim=1)
    elif method == "sum":
        x = x[:, 1:].sum(dim=1)
    elif method == "joint":
        x = torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=1)
    return x


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
    
    cfg.data.module.train_transforms = cfg.data.module.test_transforms

    # instantiate the model using this config
    model = instantiate(cfg.model)
    model.eval()
    model.to("cuda")
    # instantiate the dataset from the config
    datamodule = instantiate(cfg.data).module
    datamodule.setup()

    # Sample batch
    b = datamodule.val_dataset[0][0]
    assert b.shape[1] == b.shape[2]
    imgsize = b.shape[2]

    # Infer feature size
    f = model.extract_features(torch.randn(1, *b.shape).to("cuda"))
    f = aggregate_features(f, method=cfg.model.token_aggregation_method)
    feature_dim = f.shape[-1]

    for split in (("train", "test", "val") if not cfg.precomputed_features.split else [cfg.precomputed_features.split]):
        if split == "train":
            dataset = datamodule.train_dataset
            dataloader = datamodule.train_dataloader()
        elif split == "test":
            try:
                dataset = datamodule.test_dataset
                dataloader = datamodule.test_dataloader()
            except AttributeError:
                print(f"No test dataset available for {datamodule.__class__.__name__}")
                continue
        elif split == "val":
            dataset = datamodule.val_dataset
            dataloader = datamodule.val_dataloader()
        
        fname = FNAME_FORMAT_FEATURES.format(
            model=cfg.model.type.replace('/', '_').replace('.', '_'),
            dataset=cfg.data.module.name,
            split=split,
            imgsize=imgsize,
            precision=cfg.precomputed_features.precision,
        )
        
        out_dir = Path(cfg.data_dir) / "precomputed_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / fname
        print(f"Saving results to {str(out_file)}")
        
        with h5py.File(out_file, "w") as f:
            num_samples = len(dataset)
            print(f"{num_samples = }")

            dset_features = f.create_dataset(
                "features",
                shape=(num_samples, feature_dim),
                dtype=f"float{cfg.precomputed_features.precision}",
                chunks=(1, feature_dim),
                compression=cfg.precomputed_features.compression,
            )
            dset_labels = f.create_dataset(
                "labels",
                shape=(num_samples,),
                dtype="int64",
            )

            index = 0
            for batch in tqdm(dataloader, desc=f"{split.upper()} Batches"):
                x, y = batch
                x = x.to("cuda")
                features = aggregate_features(
                    model.extract_features(x), method=cfg.model.token_aggregation_method
                ).detach().cpu().numpy()
                batch_size = len(y)
                # Shape: (batch_size, feature_dim)
                dset_features[index : index + batch_size] = features
                dset_labels[index : index + batch_size] = y.numpy()
                index += batch_size


if __name__ == "__main__":
    make_omegaconf_resolvers()
    extract_features_hdf5()
