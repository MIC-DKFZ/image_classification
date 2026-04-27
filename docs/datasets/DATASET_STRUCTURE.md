# Dataset Structure And Split Conventions

This document describes the dataset structure expected by the current GloViTa
runtime.

## Design Principle

Datasets in the active runtime are plain PyTorch `Dataset` classes.

They are assembled centrally in:

- [../../src/glovita/datasets/factory.py](../../src/glovita/datasets/factory.py)

The goal is:

- no duplicated train/val/test orchestration per dataset
- a small number of dataset construction patterns that the factory can reuse

## The Three Main Dataset Styles

### 1. Built-In Torchvision Datasets

These do not rely on `splits.json`:

- CIFAR-10
- CIFAR-100
- ImageNet

They are built through dedicated branches in the dataset factory.

### 2. Generic Image Dataset

The runtime now also provides a reusable generic image dataset:

- `dataset="generic_image_dataset"`

This is intended for common image-dataset layouts where users should not need to
write a custom dataset class.

Supported split sources:

- `splits_json`
- `subdirs`

Supported label sources:

- `folder`
- `json`
- `csv`

Supported target types:

- scalar multiclass classification
- scalar regression

It does not currently implement generic multilabel parsing.

This path is implemented in:

- [../../src/glovita/datasets/generic_image_dataset.py](../../src/glovita/datasets/generic_image_dataset.py)

### 3. Split-Aware Repo Datasets

Most repo datasets follow a split-aware pattern:

- a plain `Dataset` class in `src/glovita/datasets/<name>.py`
- a `splits.json`
- usually a `labels.json`

These are built via `_build_generic_split_datasets(...)` in the dataset factory.

## Generic Image Dataset Layouts

### A. split file + external labels

Example:

```text
dataset_root/
├── images/
│   ├── cat/
│   │   └── cat_001.jpg
│   └── dog/
│       └── dog_001.jpg
├── splits.json
└── labels.json
```

Example `splits.json`:

```json
{
  "train": ["cat/cat_001.jpg"],
  "val": ["dog/dog_001.jpg"]
}
```

Example `labels.json`:

```json
{
  "cat/cat_001.jpg": 0,
  "dog/dog_001.jpg": 1
}
```

### B. split subdirectories + folder labels

```text
dataset_root/
└── images/
    ├── train/
    │   ├── cat/
    │   └── dog/
    ├── val/
    │   ├── cat/
    │   └── dog/
    └── test/
        ├── cat/
        └── dog/
```

In this case:

- `split_source=subdirs`
- `label_source=folder`

No `splits.json` or `labels.json` is required.

## Expected Split File Contract

For the current datasets, the standard split file looks like:

```json
{
  "train": ["sample_a", "sample_b"],
  "val": ["sample_c"],
  "test": ["sample_d"]
}
```

Important points:

- the exact sample IDs are dataset-specific
- some datasets use image-relative paths
- some datasets use sample IDs that the dataset class resolves internally

If `test` is missing:

- the dataset factory reuses the validation dataset as the test dataset

That is what currently happens for some datasets with only train/val style
splits.

## Fold-Aware Datasets

The framework already supports passing a fold identifier through:

- `data.fold`

For fold-aware datasets, `splits.json` may also contain numeric string
keys, for example:

```json
{
  "0": ["sample_a", "sample_b"],
  "1": ["sample_c", "sample_d"]
}
```

Important design point:

- the framework passes the fold value through
- the dataset implementation defines what that fold means
- the framework does not impose a universal fold interpretation

This is deliberate. Different datasets may need:

- precomputed folds
- patient-level folds
- site-level folds
- custom train/val definitions per fold

## labels.json / labels.csv

Most custom datasets also use a `labels.json` file, typically of the form:

```json
{
  "sample_a": 0,
  "sample_b": 1
}
```

Again, the exact key convention is dataset-specific. The dataset class owns the
mapping logic.

The generic image dataset also supports a CSV labels file. The relevant config
fields are:

- `path_column`
- `label_column`

## Recommended Dataset Directory Layout

The runtime does not enforce one single universal directory tree, but the
following pattern is recommended for split-aware custom datasets:

```text
<dataset_root>/
├── images/                # or any dataset-specific raw-data directory
├── splits.json
├── labels.json
└── class_map.json         # optional helper file
```

The important part is not the exact folder name. The important part is that the
dataset class and split file agree on how to resolve sample IDs.

## How The Dataset Factory Uses The Split Files

The generic path in [factory.py](../../src/glovita/datasets/factory.py) does the following:

1. resolve train/test transforms
2. instantiate the dataset class with:
   - `split="train"`
   - `split="val"`
   - optionally `split="test"`
3. pass `split_file="splits.json"` by default
4. check whether a `test` split actually exists
5. build dataloaders

So the dataset class contract is simple:

- accept `split`
- accept `transform`
- usually accept `split_file`
- interpret the split keys for that dataset

## Adding A New Dataset

For a new dataset, decide first whether the generic path is enough.

### Reuse The Generic Image Dataset When:

- samples are ordinary image files
- labels are scalar targets
- splits come from a split file or split subdirectories
- labels come from folder names, JSON, or CSV

### Write A Custom Dataset Class When:

- the dataset has custom metadata resolution
- you need grouped or patient-level logic
- you decode videos or volumes
- one sample contains multiple files
- targets are more complex than one scalar per sample

For a new custom split-aware dataset:

1. Add a config class in [../../src/glovita/configs/data.py](../../src/glovita/configs/data.py)
2. Add a dataset class in `src/glovita/datasets/<name>.py`
3. Add a factory entry in [../../src/glovita/datasets/factory.py](../../src/glovita/datasets/factory.py)
4. Add augmentation defaults and, if needed, policy implementations

If your dataset follows the standard split-aware pattern, try to reuse:

- `_build_generic_split_datasets(...)`

rather than writing a new custom builder.

## Dataset-Specific Constructor Args

If a dataset needs special constructor arguments that are not part of the shared
data schema, use:

- `data.dataset_kwargs`

Example:

```bash
glovita_train \
  --data.dataset your_dataset \
  --data.data_root_dir /data/YourDataset \
  --data.dataset_kwargs.some_flag true
```

This is the same escape-hatch design used elsewhere in the project:

- keep common parameters explicit
- use typed `*_kwargs` only for the rare family-specific extras

## Notes On Existing Docs

For runtime behavior, the source of truth is:

- [../../src/glovita/configs/data.py](../../src/glovita/configs/data.py)
- [../../src/glovita/datasets/factory.py](../../src/glovita/datasets/factory.py)
- the dataset class itself
