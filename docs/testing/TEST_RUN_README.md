# Smoke Test Notes

This file gives short smoke-test examples for the current runtime.

For the full user-facing explanation of the config tree and CLI model, start
with:

- [../../README.md](../../README.md)

## Current Entrypoints

Use:

- `glovita_train` for training
- `glovita_infer` for checkpoint-based evaluation
- `glovita_extract_features` for feature extraction

The CLI is generated from the typed config schema in:

- [../../src/glovita/configs](../../src/glovita/configs)

## Minimal Smoke Test

```bash
glovita_train \
  --training.epochs 1 \
  --dataloading.batch_size 8 \
  data:cifar10-config --data.data_root_dir ./data \
  model.encoder:timm-encoder-config --model.encoder.type resnet18 --model.encoder.no_pretrained \
  model.head:classification-head-config
```

## What A Smoke Test Should Validate

For a fast runtime check, verify that:

- config parsing works
- dataset paths resolve
- augmentations build
- model and PEFT assembly succeeds
- one short train/validation cycle completes
- run metadata and checkpoints are written

## Useful Variants

### Check MLflow Path

```bash
glovita_train \
  --training.epochs 1 \
  --dataloading.batch_size 8 \
  data:cifar10-config --data.data_root_dir ./data \
  model.encoder:timm-encoder-config --model.encoder.type resnet18 --model.encoder.no_pretrained \
  model.head:classification-head-config \
  logger:mlflow-logger-config
```

### Disable External Logging Entirely

```bash
glovita_train \
  --training.epochs 1 \
  --dataloading.batch_size 8 \
  --data.dataset cifar10 \
  --data.data_root_dir ./data \
  --model.encoder.encoder_type timm \
  --model.encoder.type resnet18 \
  --model.encoder.no_pretrained \
  --model.head.head_type classification \
  logger:no-logger-config
```

## Related Docs

- [../../README.md](../../README.md): main usage guide
- [../../tests/README.md](../../tests/README.md): test suite overview
