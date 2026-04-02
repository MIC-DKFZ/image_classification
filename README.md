# Classification Downstream

This repository contains a classification/regression training stack built on:

- PyTorch
- TorchMetrics
- Accelerate
- Weights & Biases
- Pydantic
- Tyro

The current codebase does not use Hydra or Lightning in the active runtime path.

## Current Architecture

The runtime is organized around a few central entrypoints:

- [train.py](/home/s522r/Desktop/classification_downstream/train.py): training entrypoint
- [infer.py](/home/s522r/Desktop/classification_downstream/infer.py): inference / evaluation entrypoint
- [src/configs](/home/s522r/Desktop/classification_downstream/src/configs): user-facing schema and defaults
- [datasets/factory.py](/home/s522r/Desktop/classification_downstream/datasets/factory.py): dataset and dataloader assembly
- [models/factory.py](/home/s522r/Desktop/classification_downstream/models/factory.py): encoder + head assembly
- [augmentation/policies/registry.py](/home/s522r/Desktop/classification_downstream/augmentation/policies/registry.py): augmentation policy resolution

Important runtime conventions:

- Dataset defaults live in [src/configs/data.py](/home/s522r/Desktop/classification_downstream/src/configs/data.py).
- User-facing augmentation knobs live in [src/configs/augmentation.py](/home/s522r/Desktop/classification_downstream/src/configs/augmentation.py).
- User-facing dataloader knobs live in [src/configs/dataloading.py](/home/s522r/Desktop/classification_downstream/src/configs/dataloading.py).
- Augmentation implementations live under [augmentation/policies](/home/s522r/Desktop/classification_downstream/augmentation/policies).
- Model implementations live under [models](/home/s522r/Desktop/classification_downstream/models).

## Installation

Install the project requirements in an environment that already has a compatible PyTorch build for your machine:

```bash
pip install -r requirements.txt
```

You may need to install `torch`, `torchvision`, and `torchaudio` manually from the appropriate PyTorch CUDA index for your system.

## How To Run

Tyro generates the CLI directly from the pydantic config schema.

Example with discriminated-union subcommands:

```bash
python train.py \
  --dataloading.batch-size 128 \
  data:cifar10-config --data.data-root-dir ./data \
  model.encoder:timm-encoder-config --model.encoder.type resnet50.a1_in1k --model.encoder.no-pretrained \
  model.head:classification-head-config \
  peft:full-finetuning-config
```

Example without subcommand shorthand:

```bash
python train.py \
  --data.dataset cifar10 \
  --data.data-root-dir ./data \
  --model.encoder.encoder-type timm \
  --model.encoder.type resnet50.a1_in1k \
  --model.encoder.no-pretrained \
  --model.head.head-type classification \
  --peft.method full_finetuning \
  --dataloading.batch-size 128
```

For help:

```bash
python train.py --help
python infer.py --help
```

## Config Layout

The user-facing config surface is in [src/configs](/home/s522r/Desktop/classification_downstream/src/configs):

- [root.py](/home/s522r/Desktop/classification_downstream/src/configs/root.py): top-level experiment config
- [data.py](/home/s522r/Desktop/classification_downstream/src/configs/data.py): dataset selection and dataset defaults
- [augmentation.py](/home/s522r/Desktop/classification_downstream/src/configs/augmentation.py): train/test augmentation policy selection and overrides
- [dataloading.py](/home/s522r/Desktop/classification_downstream/src/configs/dataloading.py): all `DataLoader` settings
- [model.py](/home/s522r/Desktop/classification_downstream/src/configs/model.py): encoder/head config
- [peft.py](/home/s522r/Desktop/classification_downstream/src/configs/peft.py): PEFT method config
- [optimizer.py](/home/s522r/Desktop/classification_downstream/src/configs/optimizer.py): optimizer and scheduler settings
- [training.py](/home/s522r/Desktop/classification_downstream/src/configs/training.py): trainer loop settings
- [task.py](/home/s522r/Desktop/classification_downstream/src/configs/task.py): metrics and task behavior
- [wandb_cfg.py](/home/s522r/Desktop/classification_downstream/src/configs/wandb_cfg.py): W&B settings

## Models

The current model runtime is composition-based:

- encoders in [models/encoder](/home/s522r/Desktop/classification_downstream/models/encoder)
- heads in [models/heads](/home/s522r/Desktop/classification_downstream/models/heads)
- feature aggregation in [models/feature_aggregator.py](/home/s522r/Desktop/classification_downstream/models/feature_aggregator.py)
- PEFT in [models/peft](/home/s522r/Desktop/classification_downstream/models/peft)

Available encoder families include:

- `timm`
- `torchvision`
- `transformer`
- `dinov2`
- `dinov3`
- `residual_encoder`
- `primus`

The active runtime builds the head output dimension from `config.data.num_classes` for classification tasks rather than duplicating it in the head config.

## Augmentations

Augmentations are split into:

- shared 2D defaults in [augmentation/policies/two_dim/defaults.py](/home/s522r/Desktop/classification_downstream/augmentation/policies/two_dim/defaults.py)
- shared 3D defaults in [augmentation/policies/three_dim/defaults.py](/home/s522r/Desktop/classification_downstream/augmentation/policies/three_dim/defaults.py)
- dataset-specific policies in [augmentation/policies/dataset_specific](/home/s522r/Desktop/classification_downstream/augmentation/policies/dataset_specific)

Key points:

- dataset config classes define the default `train_policy` and `test_policy`
- augmentation policy implementations define which policy names are available
- encoder preprocessing defaults can populate normalization and size parameters when the augmentation config leaves them unset
- explicit augmentation config values always override encoder-derived defaults

## Datasets And Dataloaders

Datasets are plain torch `Dataset` classes assembled via [datasets/factory.py](/home/s522r/Desktop/classification_downstream/datasets/factory.py). There is no Lightning `DataModule` in the active training path.

`build_dataloaders(...)` does the following:

1. resolve train/test transforms
2. construct train/val/test datasets
3. optionally subsample the train split with `data_fraction`
4. build PyTorch `DataLoader`s from [DataloadingConfig](/home/s522r/Desktop/classification_downstream/src/configs/dataloading.py)

## Runtime-Derived Values

Several values are intentionally derived at runtime instead of being duplicated in config:

- classification head output dimension from `data.num_classes`
- encoder preprocessing defaults (`image_size`, `resize_size`, `mean`, `std`, `patch_size`)
- effective augmentation configuration after encoder-default + user-override merging
- effective eval batch size and worker settings in dataloading
- default W&B project/group names when the user does not set them

Resolved runtime state is logged to each run directory and to W&B config for reproducibility.

## Documentation

Additional documentation is indexed in [docs/README.md](/home/s522r/Desktop/classification_downstream/docs/README.md).
