# nnFoundation Fine-Tuning

This document explains how to fine-tune the nnFoundation models (nnssl-pretrained
ResEnc and PRIMUS encoders) on 3D classification datasets.

## Runtime Path

The nnFoundation path is built from:

- [nnfoundation.py](../src/glovita/models/img_encoder/nnfoundation.py): encoder built from the checkpoint
- [generic_3d_dataset.py](../src/glovita/datasets/generic_3d_dataset.py): generic blosc2 volume dataset
- [three_dim/defaults.py](../src/glovita/augmentation/policies/three_dim/defaults.py): `default_nnfoundation` policies
- [trainer.py](../src/glovita/training/trainer.py): per-epoch step caps and final validation

## Encoder

Select the encoder with the `model.encoder:nnfoundation_encoder_config`
subcommand and `--model.encoder.checkpoint_path /path/to/checkpoint.pth`.

The architecture is read from the `nnssl_adaptation_plan` stored in the
checkpoint and built like nnU-Net's `get_network_from_plans`. Supported
architectures:

- `ResEncL` preset and `ResidualEncoderUNet` plans
- `Primus` / `PrimusX` plans and the `PrimusS` / `PrimusB` / `PrimusM` / `PrimusL` presets

Weight loading follows nnU-Net's `load_pretrained_weights`, restricted to the
stem and encoder keys of the plan. The decoder is dropped.

Options:

- `--model.encoder.input_channels`: number of input channels. If it exceeds the
  pretraining channels, the pretrained input projection is repeated and rescaled.
- `--model.encoder.no_pretrained`: use only the architecture from the checkpoint.
- `--model.encoder.drop_path_rate`: PRIMUS stochastic depth. Defaults to the
  value in the checkpoint plan.
- `--model.encoder.input_shape`: input patch size. Defaults to the
  recommended downstream patch size from the checkpoint plan.

The encoder returns average-pooled features: the mean over all patch tokens for
PRIMUS and the spatial mean of the last stage for ResEnc.

The recommended downstream patch size from the adaptation plan is used as the
default augmentation `patch_size`.

## Dataset

Select the dataset with the `data:generic3d_dataset_config` subcommand. Expected layout:

```text
dataset_root/
├── images/
│   ├── case_001.b2nd
│   └── ...
├── dataset.json
├── labels.json
└── splits.json
```

`dataset.json`:

```json
{
  "num_classes": 3,
  "subtask": "multiclass"
}
```

`num_classes` and `subtask` (`multiclass` or `multilabel`) fill the data config
unless they are set explicitly on the CLI.

`labels.json` maps case ids to labels:

- multiclass: one-hot list (argmax is used) or integer class index
- multilabel: multi-hot list

`splits.json` is either a single split or a list with one split per fold:

```json
[
  {"train": [...], "val": [...], "test": [...]},
  {"train": [...], "val": [...], "test": [...]}
]
```

The fold is selected with `--data.fold` (default: fold 0, or every fold up to
`--training.cv_folds`). The `test` split is optional for training and is used by
`glovita_infer`.

Images are expected to be preprocessed channel-first volumes; no intensity
normalization is applied at load time.

## Augmentation

`generic_3d_dataset` defaults to:

- train: `default_nnfoundation` (random crop, rotation, scaling, noise, blur,
  brightness, contrast, low resolution, gamma, mirroring)
- val / test: `default_nnfoundation` (center crop / pad only)

## Training

Epochs can be capped like Lightning's `limit_train_batches` / `limit_val_batches`:

- `--training.train_steps_per_epoch`: optimizer steps per epoch
  (micro-batches = steps x `gradient_accumulation_steps`)
- `--training.val_steps_per_epoch`: validation batches per process and epoch

If `val_steps_per_epoch` is set, `best.pt` is evaluated on the full validation
set after training. The results are logged with a `final_` prefix and written to
`final_val_metrics.json` in the run directory.

`--training.ddp_find_unused_parameters` enables DDP's unused-parameter detection
if a model leaves parameters without gradients.

## Example

Full fine-tuning on 4 GPUs (effective batch size 1 x 8 x 4 = 32).
Global options come first; each subcommand is followed by its own options:

```bash
accelerate launch --multi_gpu --num_processes 4 -m glovita.cli.train \
  --task.metrics f1 balanced_acc ap auroc pr \
  --optimizer.name AdamW \
  --optimizer.lr 1e-4 \
  --optimizer.weight_decay 0.01 \
  --optimizer.warmup_epochs 10 \
  --training.epochs 100 \
  --training.precision fp16 \
  --training.gradient_accumulation_steps 8 \
  --training.train_steps_per_epoch 50 \
  --training.val_steps_per_epoch 100 \
  --training.best_checkpoint_metric val_AUROC \
  --dataloading.batch_size 1 \
  --dataloading.num_workers 8 \
  data:generic3d_dataset_config \
  --data.data_root_dir /path/to/dataset \
  model.encoder:nnfoundation_encoder_config \
  --model.encoder.checkpoint_path /path/to/checkpoint.pth \
  --model.encoder.input_channels 1 \
  model.head:classification_head_config \
  --model.head.init_std 0.01 \
  logger:mlflow_logger_config \
  --logger.experiment_name <experiment_name> \
  --logger.run_name <run_name>
```

Evaluate the best checkpoint on the test split and store per-case outputs:

```bash
glovita_infer \
  --exp_dir experiments/generic_3d_dataset/<group> \
  --checkpoint best \
  --dataloading.batch_size 1 \
  --pred_output predictions.pt \
  data:generic3d_dataset_config \
  --data.data_root_dir /path/to/dataset
```

`predictions.pt` contains `preds`, `labels`, `probs`, and `case_ids`.
