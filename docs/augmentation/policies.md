# Augmentation Policies

## Current Layout

Augmentation behavior is split into three layers:

1. user-facing defaults and overrides in [augmentation.py](/home/s522r/Desktop/classification_downstream/src/glovita/configs/augmentation.py)
2. dataset default policy selection in [data.py](/home/s522r/Desktop/classification_downstream/src/glovita/configs/data.py)
3. policy implementations in [policies](/home/s522r/Desktop/classification_downstream/src/glovita/augmentation/policies)

Implementation folders:

- shared 2D policies: [two_dim](/home/s522r/Desktop/classification_downstream/src/glovita/augmentation/policies/two_dim)
- shared 3D policies: [three_dim](/home/s522r/Desktop/classification_downstream/src/glovita/augmentation/policies/three_dim)
- dataset-specific policies: [dataset_specific](/home/s522r/Desktop/classification_downstream/src/glovita/augmentation/policies/dataset_specific)

## Shared Policies

2D train policies are defined in [defaults.py](/home/s522r/Desktop/classification_downstream/src/glovita/augmentation/policies/two_dim/defaults.py):

- `default_2d_1` to `default_2d_5`
- `default_2d_randaugment`

2D test policy:

- `shared_default_2d`

3D train policies are defined in [defaults.py](/home/s522r/Desktop/classification_downstream/src/glovita/augmentation/policies/three_dim/defaults.py):

- `default_3d_1` to `default_3d_4`
- `default_nnunet`
- `default_nnunet_DA5`

3D test policy:

- `shared_default_3d`

## Dataset Defaults

Dataset-specific default train/test policy names live in the dataset config classes in [data.py](/home/s522r/Desktop/classification_downstream/src/glovita/configs/data.py), not in the augmentation modules.

Examples:

- `Cifar10Config`: `train_policy="randaugment"`, `test_policy="default"`
- `ImagenetConfig`: `train_policy="randaugment_448"`, `test_policy="default_448"`
- `ChestXRay14Config`: `train_policy="default_2d_2"`, `test_policy="shared_default_2d"`

## Runtime Resolution

The active runtime path is:

1. dataset config selects default train/test policy names
2. user may override them via CLI:
   - `--data.augmentation.train-policy ...`
   - `--data.augmentation.test-policy ...`
3. [registry.py](/home/s522r/Desktop/classification_downstream/src/glovita/augmentation/policies/registry.py) resolves those names to actual builders
4. [factory.py](/home/s522r/Desktop/classification_downstream/src/glovita/datasets/factory.py) merges encoder preprocessing defaults with explicit augmentation overrides

## CLI Examples

```bash
python train.py \
  --data.dataset cifar10 \
  --data.data-root-dir ./data \
  --data.augmentation.train-policy randaugment \
  --data.augmentation.test-policy default
```

```bash
python train.py \
  --data.dataset chestxray14 \
  --data.data-root-dir ./data \
  --data.augmentation.train-policy default_2d_3 \
  --data.augmentation.test-policy shared_default_2d \
  --data.augmentation.image-size 256 \
  --data.augmentation.resize-size 320
```

## Extending Policies

To add a new dataset-specific policy:

1. create or update `augmentation/policies/dataset_specific/<dataset>.py`
2. export:
   - `SPATIAL_DIM`
   - `TRAIN_POLICIES`
   - `TEST_POLICIES`
3. set the dataset defaults in the matching config class in [data.py](/home/s522r/Desktop/classification_downstream/src/glovita/configs/data.py)

The registry does not own dataset defaults anymore; it only resolves available policy names.
