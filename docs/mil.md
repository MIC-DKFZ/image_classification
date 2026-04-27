# Multiple Instance Learning

This document explains the current MIL path in GloViTa.

## Current Scope

MIL is currently supported for:

- bags of precomputed features
- the `precomputed_features` dataset config
- the `precomputed` encoder
- the `clam` head

That means the active MIL workflow is feature-based, not raw-image-bag or
raw-video-bag MIL.

## Runtime Path

The MIL path is built from:

- [precomputed_features.py](../src/glovita/datasets/precomputed_features.py)
- [precomputed.py](../src/glovita/models/img_encoder/precomputed.py)
- [clam.py](../src/glovita/models/heads/mil/clam.py)
- [factory.py](../src/glovita/models/factory.py)
- [factory.py](../src/glovita/datasets/factory.py)

Important design point:

- CLAM consumes raw bag features directly
- it does not use the standard pooled-token feature aggregation path

This is why the model factory distinguishes between:

- normal pooled heads
- raw-feature-consuming heads

## Supported HDF5 Bag Formats

The HDF5 loader supports three layouts.

### Instance Features

```text
features: (N, D)
labels:   (N,)
```

This is not bag MIL. It is just standard feature-based classification or regression.

### Fixed-Size Bags

```text
features: (B, N, D)
labels:   (B,)
```

### Variable-Size Bags

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

## Collation Behavior

At dataloader time, MIL bags are collated into:

- `features`: padded tensor `(B, N_max, D)`
- `mask`: boolean tensor `(B, N_max)`

This padded structure is what the CLAM head receives.

## CLAM Head Configuration

The user-facing config class is:

- [ClamHeadConfig](../src/glovita/configs/model.py)

Important options:

- `variant`
  - `sb`
  - `mb`
- `gate`
- `size_arg`
- `dropout`
- `k_sample`
- `subtyping`
- `feature_prep`
- `l2_normalize_features`
- `cosine_head`
- `cosine_scale`
- `instance_eval`
- `instance_loss_weight`
- `attn_drop`
- `stochastic_topk`
- `topk_k`
- `topk_tau`
- `topk_noise_std`
- `topk_consistency_weight`

## Current CLAM Semantics

The main bag prediction is computed from dense attention over the whole bag.

Optional auxiliary behavior:

- instance-level CLAM auxiliary loss:
  - enabled by `instance_eval=True`
- top-k perturbation consistency penalty:
  - enabled by `stochastic_topk=True`

Important detail:

- the top-k branch is auxiliary
- it does not replace the main bag prediction

## Example Training Command

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
  --model.head.variant mb \
  --model.head.instance_eval \
  --model.head.stochastic_topk \
  --peft.method full_finetuning \
  --dataloading.batch_size 8
```

## Practical Notes

- `data.num_classes` must be set explicitly
- `model.encoder.feature_dim` must match the feature tensor dimension
- augmentations are not used for `precomputed_features`
- `model.feature_aggregation_method` is ignored by `clam`
- `full_finetuning` is the right PEFT config for this path unless you are
  reconstructing a checkpoint with a different PEFT method

## Current Limitations

- active MIL support is feature-based, not raw-bag image/video MIL
- inference and evaluation tooling is primarily oriented around standard
  classification use cases
- there is no separate generic MIL dataset abstraction yet

## Related Docs

- [precomputed_features.md](precomputed_features.md): HDF5 file formats and feature extraction
- [../README.md](../README.md): overall config and CLI model
