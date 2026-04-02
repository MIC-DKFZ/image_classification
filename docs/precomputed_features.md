# Precomputed Features

The current runtime supports training from precomputed HDF5 feature files using:

- dataset config: `precomputed_features`
- encoder config: `precomputed`
- the existing classification head in [models/heads/classification.py](/home/s522r/Desktop/classification_downstream/models/heads/classification.py)

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

- [datasets/precomputed_features.py](/home/s522r/Desktop/classification_downstream/datasets/precomputed_features.py)
- [models/encoder/precomputed.py](/home/s522r/Desktop/classification_downstream/models/encoder/precomputed.py)
- [datasets/factory.py](/home/s522r/Desktop/classification_downstream/datasets/factory.py)

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

[extract_cls_and_avg_patch_token.py](/home/s522r/Desktop/classification_downstream/extract_cls_and_avg_patch_token.py)
has been rewritten for the current tyro+pydantic runtime. It now uses:

- the current dataset factory
- the current composed model stack
- the current PEFT registry
- the shared feature aggregation implementation

Example extraction command:

```bash
python extract_cls_and_avg_patch_token.py \
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

Notes:

- `--method joint` reproduces the concatenated CLS-token + average patch-token representation.
- by default the extraction script applies the test/eval transform to the train split as well, so extracted train features are deterministic
- output files are written in the same `features` / `labels` HDF5 format documented above
