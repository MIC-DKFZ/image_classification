# Precomputed Features

The current runtime supports training from precomputed HDF5 feature files using:

- dataset config: `precomputed_features`
- encoder config: `precomputed`
- the existing classification head in [classification.py](/home/s522r/Desktop/classification_downstream/src/glovita/models/heads/classification.py)

This replaces the old dedicated linear model path. In the current architecture,
the `precomputed` encoder behaves like an identity backbone, so the normal head
stack still works.

## Supported File Format

Each HDF5 file must contain:

- `features`: shape `(N, D)`
- `labels`: shape `(N,)`

This matches the format used in the older extraction scripts and the filename
pattern described previously still works fine, for example:

`agg_joint_dinov2_vitb14_imagenet_train_size224_float16.h5`

The loader does not depend on the filename itself; it depends on the HDF5
datasets `features` and `labels`.

## Current Loading Path

The active runtime path is implemented in:

- [precomputed_features.py](/home/s522r/Desktop/classification_downstream/src/glovita/datasets/precomputed_features.py)
- [precomputed.py](/home/s522r/Desktop/classification_downstream/src/glovita/models/encoder/precomputed.py)
- [factory.py](/home/s522r/Desktop/classification_downstream/src/glovita/datasets/factory.py)

## Training Example

Example command:

```bash
python train.py \
  --data.dataset precomputed_features \
  --data.data-root-dir . \
  --data.num-classes 1000 \
  --data.train-features-file /path/to/agg_joint_dinov2_vitb14_imagenet_train_size224_float16.h5 \
  --data.val-features-file /path/to/agg_joint_dinov2_vitb14_imagenet_val_size224_float16.h5 \
  --model.encoder.encoder-type precomputed \
  --model.encoder.feature-dim 1536 \
  --model.head.head-type classification \
  --peft.method full_finetuning \
  --dataloading.batch-size 512
```

If you have a separate test file:

```bash
--data.test-features-file /path/to/agg_joint_dinov2_vitb14_imagenet_test_size224_float16.h5
```

## Notes

- `data.data_root_dir` is still required by the shared data schema, but it is not
  used for feature loading beyond normal config consistency.
- `data.num_classes` must currently be set explicitly in the config/CLI.
- `model.encoder.feature_dim` must currently be set explicitly and must match the
  second dimension of the stored feature tensor.
- Augmentations are not used for `precomputed_features`.
- Classification works out of the box because the classification head only needs
  the input feature dimension and `data.num_classes`.

## Feature Extraction

[extract_features.py](/home/s522r/Desktop/classification_downstream/extract_features.py)
has been rewritten for the current tyro+pydantic runtime. It now uses:

- the current dataset factory
- the current composed model stack
- the current PEFT registry
- the shared feature aggregation implementation

It supports two extraction modes:

- explicit config mode: provide `data` + `model` and optionally `peft`
- checkpoint mode: provide `--checkpoint-path` and the script reconstructs the saved run automatically

Example extraction command:

```bash
python extract_features.py \
  --method joint \
  --output-dir ./precomputed_features \
  --data.dataset cifar10 \
  --data.data-root-dir ./data \
  --model.encoder.encoder-type timm \
  --model.encoder.type vit_base_patch16_224 \
  --model.head.head-type classification \
  --peft.method full_finetuning \
  --dataloading.batch-size 128
```

Extraction from a checkpoint saved by this repo:

```bash
python extract_features.py \
  --checkpoint-path ./experiments/cifar10/my_run/0/checkpoints/last.pt \
  --output-dir ./precomputed_features \
  --output-filename "{checkpoint}_{dataset}_{split}_{method}.h5"
```

Notes:

- `--method joint` reproduces the concatenated CLS-token + average patch-token representation.
- `peft` now defaults to `full_finetuning`, so plain backbone extraction does not require an explicit PEFT flag.
- if `--checkpoint-path` is provided, the script loads `config.json` from the checkpoint run directory and reconstructs the saved model/PEFT setup before loading the checkpoint weights.
- you can still override `data`, `model`, or `peft` explicitly on the CLI if needed.
- by default the extraction script applies the test/eval transform to the train split as well, so extracted train features are deterministic.
- output files are written in the same `features` / `labels` HDF5 format documented above.
- if `--output-filename` is unset, the script uses the default template:
  `agg_{method}_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5`
- `--output-filename` accepts a Python format string with placeholders:
  `method`, `model`, `dataset`, `split`, `imgsize`, `precision`, `checkpoint`.
