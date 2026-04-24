# Smoke Test Notes

This file gives short smoke-test examples for the current runtime.

For the full user-facing explanation of the config tree and CLI model, start
with:

- [../../README.md](../../README.md)

## Current Entrypoints

Use:

- [../../train.py](../../train.py) for training
- [../../infer.py](../../infer.py) for checkpoint-based evaluation
- [../../extract_features.py](../../extract_features.py) for feature extraction

The CLI is generated from the typed config schema in:

- [../../src/glovita/configs](../../src/glovita/configs)

## Minimal Smoke Test

```bash
python train.py \
  --training.epochs 1 \
  --dataloading.batch_size 8 \
  data:cifar10-config --data.data_root_dir ./data \
  model.encoder:timm-encoder-config --model.encoder.type resnet18 --model.encoder.no_pretrained \
  model.head:classification-head-config \
  peft:full-finetuning-config
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
python train.py \
  --training.epochs 1 \
  --dataloading.batch_size 8 \
  data:cifar10-config --data.data_root_dir ./data \
  model.encoder:timm-encoder-config --model.encoder.type resnet18 --model.encoder.no_pretrained \
  model.head:classification-head-config \
  peft:full-finetuning-config \
  logger:mlflow-logger-config
```

### Disable External Logging Entirely

```bash
python train.py \
  --training.epochs 1 \
  --dataloading.batch_size 8 \
  --data.dataset cifar10 \
  --data.data_root_dir ./data \
  --model.encoder.encoder_type timm \
  --model.encoder.type resnet18 \
  --model.encoder.no_pretrained \
  --model.head.head_type classification \
  --peft.method full_finetuning \
  logger:no-logger-config
```

## Related Docs

- [../../README.md](../../README.md): main usage guide
- [../../tests/README.md](../../tests/README.md): test suite overview
