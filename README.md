# GloViTa: Global Vision Tasks

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

- [train.py](train.py): training entrypoint
- [infer.py](infer.py): inference / evaluation entrypoint
- [src/glovita/configs](src/glovita/configs): user-facing schema and defaults
- [src/glovita/datasets/factory.py](src/glovita/datasets/factory.py): dataset and dataloader assembly
- [src/glovita/models/factory.py](src/glovita/models/factory.py): encoder + head assembly
- [src/glovita/augmentation/policies/registry.py](src/glovita/augmentation/policies/registry.py): augmentation policy resolution

Important runtime conventions:

- Dataset defaults live in [src/glovita/configs/data.py](src/glovita/configs/data.py).
- User-facing augmentation knobs live in [src/glovita/configs/augmentation.py](src/glovita/configs/augmentation.py).
- User-facing dataloader knobs live in [src/glovita/configs/dataloading.py](src/glovita/configs/dataloading.py).
- Augmentation implementations live under [src/glovita/augmentation/policies](src/glovita/augmentation/policies).
- Model implementations live under [src/glovita/models](src/glovita/models).

## Installation

Install the project requirements in an environment that already has a compatible PyTorch build for your machine:

```bash
pip install -e .
```


## How To Run

Tyro generates the CLI directly from the pydantic config schema.

Example with discriminated-union subcommands:

```bash
python train.py \
  --dataloading.batch_size 128 \
  data:cifar10-config --data.data_root_dir ./data \
  model.encoder:timm-encoder-config --model.encoder.type resnet50.a1_in1k --model.encoder.no_pretrained \
  model.head:classification-head-config \
  peft:full-finetuning-config
```

Example without subcommand shorthand:

```bash
python train.py \
  --data.dataset cifar10 \
  --data.data_root_dir ./data \
  --model.encoder.encoder_type timm \
  --model.encoder.type resnet50.a1_in1k \
  --model.encoder.no_pretrained \
  --model.head.head_type classification \
  --peft.method full_finetuning \
  --dataloading.batch_size 128
```

For help:

```bash
python train.py --help
python infer.py --help
```

Logging-only metadata can be attached without affecting the run via `--add_log.*` flags:

```bash
python train.py \
  --data.dataset cifar10 \
  --data.data_root_dir ./data \
  --model.encoder.encoder_type timm \
  --model.encoder.type resnet50.a1_in1k \
  --peft.method full_finetuning \
  --add_log.comment "baseline with larger batch size" \
  --add_log.dataset_alias CIFAR10_clean \
  --add_log.notes.run_group ablation_a
```

These values are saved in the run config and logged to W&B, but they do not affect runtime behavior.

## Config Layout

The user-facing config surface is in [src/glovita/configs](src/glovita/configs):

- [root.py](src/glovita/configs/root.py): top-level experiment config
- [data.py](src/glovita/configs/data.py): dataset selection and dataset defaults
- [augmentation.py](src/glovita/configs/augmentation.py): train/test augmentation policy selection and overrides
- [dataloading.py](src/glovita/configs/dataloading.py): all `DataLoader` settings
- [model.py](src/glovita/configs/model.py): encoder/head config
- [peft.py](src/glovita/configs/peft.py): PEFT method config
- [optimizer.py](src/glovita/configs/optimizer.py): optimizer and scheduler settings
- [training.py](src/glovita/configs/training.py): trainer loop settings
- [task.py](src/glovita/configs/task.py): metrics and task behavior
- [wandb_cfg.py](src/glovita/configs/wandb_cfg.py): W&B settings

## Models

The current model runtime is composition-based:

- encoders in [src/glovita/models/encoder](src/glovita/models/encoder)
- heads in [src/glovita/models/heads](src/glovita/models/heads)
- feature aggregation in [src/glovita/models/feature_aggregator.py](src/glovita/models/feature_aggregator.py)
- PEFT in [src/glovita/models/peft](src/glovita/models/peft)

Available encoder families include:

- `timm`
- `torchvision`
- `transformer`
- `dinov2`
- `dinov3`
- `residual_encoder`
- `primus`
- `precomputed`

Available heads include:

- `classification`
- `regression`
- `clam`

The active runtime builds the head output dimension from `config.data.num_classes` for classification tasks rather than duplicating it in the head config.

## Augmentations

Augmentations are split into:

- shared 2D defaults in [src/glovita/augmentation/policies/two_dim/defaults.py](src/glovita/augmentation/policies/two_dim/defaults.py)
- shared 3D defaults in [src/glovita/augmentation/policies/three_dim/defaults.py](src/glovita/augmentation/policies/three_dim/defaults.py)
- dataset-specific policies in [src/glovita/augmentation/policies/dataset_specific](src/glovita/augmentation/policies/dataset_specific)

Key points:

- dataset config classes define the default `train_policy` and `test_policy`
- augmentation policy implementations define which policy names are available
- encoder preprocessing defaults can populate normalization and size parameters when the augmentation config leaves them unset
- explicit augmentation config values always override encoder-derived defaults

## Datasets And Dataloaders

Datasets are plain torch `Dataset` classes assembled via [src/glovita/datasets/factory.py](src/glovita/datasets/factory.py). There is no Lightning `DataModule` in the active training path.

`build_dataloaders(...)` does the following:

1. resolve train/test transforms
2. construct train/val/test datasets
3. optionally subsample the train split with `data_fraction`
4. build PyTorch `DataLoader`s from [DataloadingConfig](src/glovita/configs/dataloading.py)

Split membership comes from the dataset implementation. For the current
datasets this is typically `splits.json` with `train` / `val` / `test` keys.
Future fold-aware datasets can also use numeric keys such as `0`, `1`, ...
when their dataset class interprets those keys explicitly.

For `precomputed_features`, the runtime supports:

- plain instance features `(N, D)`
- fixed-size bags `(B, N, D)`
- variable-size bags stored as concatenated features plus `bag_ptr` or `bag_lengths`

Bag-style precomputed inputs are collated into padded `{features, mask}` batches
and are intended for MIL heads such as `clam`.

## Runtime-Derived Values

Several values are intentionally derived at runtime instead of being duplicated in config:

- classification head output dimension from `data.num_classes`
- encoder preprocessing defaults (`image_size`, `resize_size`, `mean`, `std`, `patch_size`)
- effective augmentation configuration after encoder-default + user-override merging
- effective eval batch size and worker settings in dataloading
- default W&B project/group names when the user does not set them
- cross-validation behavior:
  - `training.cv_folds > 1` runs folds sequentially as `0..N-1`
  - `data.fold` forces a single fold run
  - dataset-specific fold-key interpretation is handled by the dataset implementation

Resolved runtime state is logged to each run directory and to W&B config for reproducibility.

## Documentation

Additional documentation is indexed in [docs/README.md](docs/README.md).
