# Augmentation Policies

This document explains how augmentation is organized in the current GloViTa
runtime, how policy selection works from the CLI, and how to add new shared or
dataset-specific policies.

## Design Intent

Augmentation is split into three layers on purpose:

1. user-facing override surface in [configs/augmentation.py](../../src/glovita/configs/augmentation.py)
2. dataset-specific defaults in [configs/data.py](../../src/glovita/configs/data.py)
3. implementation code in [src/glovita/augmentation/policies](../../src/glovita/augmentation/policies)

Why this split exists:

- users should be able to inspect defaults in config classes
- implementation details should live next to augmentation code, not in config
- dataset defaults should be easy to find without reading registry logic

## Where Things Live

Important files:

- [../../src/glovita/configs/augmentation.py](../../src/glovita/configs/augmentation.py): user-facing augmentation override block
- [../../src/glovita/configs/data.py](../../src/glovita/configs/data.py): per-dataset default `train_policy` and `test_policy`
- [../../src/glovita/augmentation/policies/registry.py](../../src/glovita/augmentation/policies/registry.py): resolves policy names to builders
- [../../src/glovita/augmentation/policies/two_dim/defaults.py](../../src/glovita/augmentation/policies/two_dim/defaults.py): shared 2D policies
- [../../src/glovita/augmentation/policies/three_dim/defaults.py](../../src/glovita/augmentation/policies/three_dim/defaults.py): shared 3D policies
- [../../src/glovita/augmentation/policies/dataset_specific](../../src/glovita/augmentation/policies/dataset_specific): dataset-specific policy modules

## User-Facing Config

The augmentation config block is:

- `data.augmentation`

Key fields:

- `train_policy`
- `test_policy`
- `image_size`
- `resize_size`
- `crop_size`
- `cutout_size`
- `patch_size`
- `mean`
- `std`
- `train_kwargs`
- `test_kwargs`

Runtime behavior:

- dataset config provides default train/test policy names
- encoder preprocessing may fill in unset size/normalization fields
- explicit augmentation values always override encoder-derived defaults

## Shared Policy Names

### Shared 2D Train Policies

Defined in [two_dim/defaults.py](../../src/glovita/augmentation/policies/two_dim/defaults.py):

- `default_2d_1`
- `default_2d_2`
- `default_2d_3`
- `default_2d_4`
- `default_2d_5`
- `default_2d_randaugment`

The intent is progressive strength:

- `default_2d_1`: close to evaluation-style preprocessing
- `default_2d_2`: mild geometric augmentation
- `default_2d_3`: moderate geometry + color
- `default_2d_4`: stronger and broader perturbations
- `default_2d_5`: the most aggressive shared 2D default

### Shared 2D Test Policy

- `shared_default_2d`

### Shared 3D Train Policies

Defined in [three_dim/defaults.py](../../src/glovita/augmentation/policies/three_dim/defaults.py):

- `default_3d_1`
- `default_3d_2`
- `default_3d_3`
- `default_3d_4`
- `default_nnunet`
- `default_nnunet_DA5`

### Shared 3D Test Policy

- `shared_default_3d`

## Dataset-Specific Defaults

Dataset defaults live in [configs/data.py](../../src/glovita/configs/data.py), not in the augmentation registry.

Examples:

- `Cifar10Config`
  - `train_policy="randaugment"`
  - `test_policy="default"`
- `ImagenetConfig`
  - `train_policy="randaugment_448"`
  - `test_policy="default_448"`
- `ChestXRay14Config`
  - `train_policy="default_2d_2"`
  - `test_policy="shared_default_2d"`

This is intentional. A user should be able to inspect dataset defaults without
reading augmentation implementation code.

## Runtime Resolution

At runtime, the path is:

1. dataset config provides default policy names
2. CLI may override them via `data.augmentation.*`
3. the augmentation registry validates and resolves the names
4. dataset factory merges:
   - policy defaults
   - encoder preprocessing defaults
   - explicit augmentation overrides

That merge happens in:

- [../../src/glovita/datasets/factory.py](../../src/glovita/datasets/factory.py)

## CLI Examples

### Override Only The Train Policy

```bash
python train.py \
  --data.dataset chestxray14 \
  --data.data_root_dir /data/ChestXray14 \
  --data.augmentation.train_policy default_2d_4
```

### Override Both Policies And Sizes

```bash
python train.py \
  --data.dataset imagenet \
  --data.data_root_dir /data/ILSVRC \
  --data.augmentation.train_policy randaugment_224 \
  --data.augmentation.test_policy default_224 \
  --data.augmentation.image_size 224 \
  --data.augmentation.resize_size 256
```

### Pass Policy-Specific Escape-Hatch Values

```bash
python train.py \
  --data.dataset cifar10 \
  --data.data_root_dir ./data \
  --data.augmentation.train_kwargs.cutout_size 12
```

Use `train_kwargs` and `test_kwargs` only for genuinely policy-specific values
that are not worth promoting into the shared schema.

## Adding A New Dataset-Specific Policy

To add a dataset-specific policy module:

1. Create or update:
   - `src/glovita/augmentation/policies/dataset_specific/<dataset>.py`
2. Export:
   - `SPATIAL_DIM`
   - `TRAIN_POLICIES`
   - `TEST_POLICIES`
3. Set dataset defaults in the matching config class in:
   - [configs/data.py](../../src/glovita/configs/data.py)

The registry does not own dataset defaults. It only resolves available policy
names and builders.

## Adding A New Shared Policy

For a new shared 2D or 3D policy:

1. Add the builder to:
   - [two_dim/defaults.py](../../src/glovita/augmentation/policies/two_dim/defaults.py)
   - or [three_dim/defaults.py](../../src/glovita/augmentation/policies/three_dim/defaults.py)
2. Export it in:
   - `TRAIN_POLICIES`
   - or `TEST_POLICIES`
3. Add the policy name to the type alias in:
   - [configs/augmentation.py](../../src/glovita/configs/augmentation.py)

That last step matters because policy names are part of the typed user-facing
config surface.
