# Multiple Instance Learning

This document describes the current MIL path in GloViTa.

## Current Scope

MIL is currently supported through:

- dataset config: `precomputed_features`
- encoder config: `precomputed`
- head config: `clam`

This means the active MIL path expects bags of precomputed features rather than
raw images or raw videos.

## Supported Bag Formats

The HDF5 loader in [precomputed_features.py](../src/glovita/datasets/precomputed_features.py)
supports:

- fixed-size bags:
  - `features`: shape `(B, N, D)`
  - `labels`: shape `(B,)`
- variable-size bags:
  - `features`: shape `(M, D)`
  - `labels`: shape `(B,)`
  - plus either `bag_ptr` or `bag_lengths`

At dataloader time, bags are collated into:

- `features`: padded tensor `(B, N_max, D)`
- `mask`: boolean tensor `(B, N_max)`

## CLAM Head

The active CLAM implementation lives in [clam.py](../src/glovita/models/heads/mil/clam.py).

Supported variants:

- `variant=sb`
- `variant=mb`

Important config knobs:

- `gate`
- `size_arg`
- `dropout`
- `k_sample`
- `subtyping`
- `feature_prep`
- `l2_normalize_features`
- `cosine_head`
- `instance_eval`
- `instance_loss_weight`
- `attn_drop`
- `stochastic_topk`
- `topk_k`
- `topk_tau`
- `topk_noise_std`
- `topk_consistency_weight`

## Current Training Semantics

The normal CLAM bag prediction is computed from dense attention over all
instances.

Optional auxiliary losses:

- instance-level CLAM loss:
  - enabled by `instance_eval=True`
- top-k perturbation consistency loss:
  - enabled by `stochastic_topk=True`

The top-k branch does not replace the main prediction. It adds an auxiliary
penalty on top of the normal bag-level loss.

## Example

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
  --model.head.variant mb \
  --model.head.instance_eval \
  --model.head.stochastic_topk \
  --peft.method full_finetuning
```

## Limitations

- the active MIL path is based on precomputed features, not raw image/video bags
- there is no separate MIL-specific dataset abstraction yet
- inference / evaluation tooling is still mostly written with standard
  classification in mind
