# Precomputed Features

The current runtime supports training from precomputed HDF5 feature files using:

- dataset config: `precomputed_features`
- encoder config: `precomputed`
- either the standard classification head in [classification.py](../src/glovita/models/heads/classification.py)
- or the MIL `clam` head in [clam.py](../src/glovita/models/heads/mil/clam.py)

This replaces the old dedicated linear-model path. The `precomputed` encoder is
an identity backbone, so the rest of the model stack still works normally.

## Supported File Formats

Each HDF5 file must contain `features` and `labels`. The loader supports three
layouts:

- instance features:
  - `features`: shape `(N, D)`
  - `labels`: shape `(N,)`
- fixed-size bags:
  - `features`: shape `(B, N, D)`
  - `labels`: shape `(B,)`
- variable-size bags:
  - `features`: shape `(M, D)`
  - `labels`: shape `(B,)`
  - plus either:
    - `bag_ptr`: shape `(B + 1,)`
    - or `bag_lengths`: shape `(B,)`

The default dataset keys can be overridden via `--data.dataset_kwargs.*` if your
file uses different names.

## Current Loading Path

The active runtime path is implemented in:

- [precomputed_features.py](../src/glovita/datasets/precomputed_features.py)
- [precomputed.py](../src/glovita/models/encoder/precomputed.py)
- [factory.py](../src/glovita/datasets/factory.py)

For bag files, the dataloader pads bags within a batch and passes CLAM a
dictionary with:

- `features`: padded tensor of shape `(B, N_max, D)`
- `mask`: boolean tensor of shape `(B, N_max)`

## Training Examples

Standard classification on precomputed instance features:

```bash
python train.py \
  --data.dataset precomputed_features \
  --data.data_root_dir . \
  --data.num_classes 1000 \
  --data.train_features_file /path/to/train_features.h5 \
  --data.val_features_file /path/to/val_features.h5 \
  --model.encoder.encoder_type precomputed \
  --model.encoder.feature_dim 1536 \
  --model.head.head_type classification \
  --peft.method full_finetuning \
  --dataloading.batch_size 512
```

MIL with bag features and CLAM:

```bash
python train.py \
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
  --peft.method full_finetuning \
  --dataloading.batch_size 8
```

If you have a separate test file:

```bash
--data.test_features_file /path/to/test_features.h5
```

## Notes

- `data.data_root_dir` is still required by the shared data schema, but it is
  not used for feature loading beyond normal config consistency.
- `data.num_classes` must be set explicitly.
- `model.encoder.feature_dim` must be set explicitly and must match the last
  dimension of the stored feature tensor.
- Augmentations are not used for `precomputed_features`.
- `clam` consumes raw bag features directly, so
  `model.feature_aggregation_method` is ignored for that head.
- Bag-style precomputed inputs are intended for MIL heads. Standard pooled heads
  will raise if they receive padded bag batches.

## Feature Extraction

[extract_features.py](../extract_features.py) writes the same `features` /
`labels` HDF5 format and supports two modes:

- explicit config mode: provide `data` + `model` and optionally `peft`
- checkpoint mode: provide `--checkpoint_path` and the script reconstructs the
  saved run automatically

Example extraction command:

```bash
python extract_features.py \
  --method joint \
  --output_dir ./precomputed_features \
  --data.dataset cifar10 \
  --data.data_root_dir ./data \
  --model.encoder.encoder_type timm \
  --model.encoder.type vit_base_patch16_224 \
  --model.head.head_type classification \
  --peft.method full_finetuning \
  --dataloading.batch_size 128
```

Extraction from a checkpoint saved by this repo:

```bash
python extract_features.py \
  --checkpoint_path ./experiments/cifar10/my_run/0/checkpoints/last.pt \
  --output_dir ./precomputed_features \
  --output_filename "{checkpoint}_{dataset}_{split}_{method}.h5"
```

Notes:

- `--method joint` reproduces the concatenated CLS-token + average patch-token representation.
- `peft` defaults to `full_finetuning`, so plain backbone extraction does not
  require an explicit PEFT flag.
- if `--checkpoint_path` is provided, the script loads `config.json` from the
  checkpoint run directory and reconstructs the saved model/PEFT setup before
  loading the checkpoint weights.
- by default the extraction script applies the test/eval transform to the train
  split as well, so extracted train features are deterministic.
- if `--output_filename` is unset, the script uses the default template:
  `agg_{method}_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5`
- `--output_filename` accepts a Python format string with placeholders:
  `method`, `model`, `dataset`, `split`, `imgsize`, `precision`, `checkpoint`.
