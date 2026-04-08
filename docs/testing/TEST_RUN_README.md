# Test Run Notes

This repository no longer uses the old `main.py` + Hydra command style described
in earlier versions of this document.

## Current Runtime

Use:

- [train.py](../../train.py) for training
- [infer.py](../../infer.py) for evaluation

with tyro-generated CLI arguments from the pydantic config schema in
[src/glovita/configs](../../src/glovita/configs).

## Example Smoke Test

```bash
python train.py \
  --training.epochs 1 \
  --dataloading.batch-size 8 \
  data:cifar10-config --data.data-root-dir ./data \
  model.encoder:timm-encoder-config --model.encoder.type resnet18 --model.encoder.no-pretrained \
  model.head:classification-head-config \
  peft:full-finetuning-config
```

## What To Validate

For quick runtime validation, verify:

- config parsing works
- dataset paths resolve correctly
- augmentations are built successfully
- model + PEFT assembly runs
- one short train/validation cycle completes

For up-to-date testing guidance, prefer [TESTING.md](TESTING.md) and the top-level [README.md](../../README.md).
