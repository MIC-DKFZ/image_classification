# Precomputed Features

This document explains how GloViTa handles training from precomputed HDF5
feature files and how `glovita_extract_features` produces those files.

## Supported Runtime Path

Training from precomputed features uses:

- dataset config: `precomputed_features`
- encoder config: `precomputed`
- a standard head such as `classification`
- or the MIL `clam` head for bag data

The `precomputed` encoder is effectively an identity backbone. This lets the
rest of the normal model stack keep working:

- model config
- head selection
- PEFT reconstruction
- trainer logic

## Supported HDF5 Layouts

Each HDF5 file must contain:

- `features`
- `labels`

The current loader supports three shapes.

### 1. Instance Features

```text
features: (N, D)
labels:   (N,)
```

### 2. Fixed-Size Bags

```text
features: (B, N, D)
labels:   (B,)
```

### 3. Variable-Size Bags

```text
features:    (M, D)
labels:      (B,)
bag_ptr:     (B + 1,)
```

or

```text
features:    (M, D)
labels:      (B,)
bag_lengths: (B,)
```

## Loader Behavior

The active loader lives in:

- [../src/glovita/datasets/precomputed_features.py](../src/glovita/datasets/precomputed_features.py)

The dataset factory integrates it through:

- [../src/glovita/datasets/factory.py](../src/glovita/datasets/factory.py)

For bag-style data, the dataloader pads bags within a batch and returns:

- `features`: padded tensor `(B, N_max, D)`
- `mask`: boolean tensor `(B, N_max)`

This is intended for bag-aware heads such as `clam`.

## Training Examples

### Standard Classification On Feature Files

```bash
glovita_train \
  --data.dataset precomputed_features \
  --data.data_root_dir . \
  --data.num_classes 1000 \
  --data.train_features_file /path/to/train_features.h5 \
  --data.val_features_file /path/to/val_features.h5 \
  --model.encoder.encoder_type precomputed \
  --model.encoder.feature_dim 1536 \
  --model.head.head_type classification \
  --dataloading.batch_size 512
```

### MIL With CLAM On Bag Files

```bash
glovita_train \
  --data.dataset precomputed_features \
  --data.data_root_dir . \
  --data.num_classes 2 \
  --data.train_features_file /path/to/train_bags.h5 \
  --data.val_features_file /path/to/val_bags.h5 \
  --model.encoder.encoder_type precomputed \
  --model.encoder.feature_dim 1024 \
  --model.head.head_type clam \
  --model.head.variant sb \
  --model.head.instance_eval \
  --dataloading.batch_size 8
```

If a separate test file exists:

```bash
--data.test_features_file /path/to/test_features.h5
```

## Important Notes

- `data.data_root_dir` is part of the shared data schema but is not used
  for feature loading in the same way as image datasets
- `data.num_classes` must be set explicitly
- `model.encoder.feature_dim` must match the stored feature dimension
- augmentations are not used for `precomputed_features`
- `clam` consumes raw bag features directly, so `model.feature_aggregation_method`
  is ignored for that head

## Feature Extraction

`glovita_extract_features` writes the same HDF5 format.

It supports:

- explicit config mode
- checkpoint reconstruction mode

### Explicit Config Mode

```bash
glovita_extract_features \
  --method joint \
  --output_dir ./precomputed_features \
  --data.dataset cifar10 \
  --data.data_root_dir ./data \
  --model.encoder.encoder_type timm \
  --model.encoder.type vit_base_patch16_224 \
  --model.head.head_type classification \
  --dataloading.batch_size 128
```

### Checkpoint Reconstruction Mode

```bash
glovita_extract_features \
  --checkpoint_path ./experiments/cifar10/my_run/0/checkpoints/last.pt \
  --output_dir ./precomputed_features \
  --output_filename "{checkpoint}_{dataset}_{split}_{method}.h5"
```

In checkpoint mode, the script:

1. finds `config.json` next to the checkpoint run directory
2. reconstructs the saved model and PEFT configuration
3. loads checkpoint weights
4. extracts features from the selected split(s)

## Extraction Options

Important extraction fields:

- `method`
  - `cls_token`
  - `avg`
  - `sum`
  - `mean_all`
  - `joint`
- `split`
  - `train`
  - `val`
  - `test`
  - or all when unset
- `precision`
- `compression`
- `output_dir`
- `output_filename`
- `use_eval_transform_for_train`

Notes:

- `joint` corresponds to concatenated CLS-token + average patch-token features
- `peft` defaults to `full_finetuning`
- by default, extraction uses evaluation transforms for the train split as well,
  so train feature extraction is deterministic

## Output Filename Template

If `--output_filename` is unset, the script uses:

```text
agg_{method}_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5
```

Supported placeholders:

- `method`
- `model`
- `dataset`
- `split`
- `imgsize`
- `precision`
- `checkpoint`

## Related Docs

- [mil.md](mil.md): MIL/CLAM path for bag features
- [../README.md](../README.md): main config and CLI model
