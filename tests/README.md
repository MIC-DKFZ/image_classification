# Test Suite

This directory contains the current pytest-based test suite for GloViTa.

## Scope

The tests primarily cover:

- dataset classes
- augmentation policy construction
- dataset factory / dataloader wiring

The test suite is useful for:

- validating refactors
- checking that config/factory integration works
- smoke-testing new dataset additions

## Current Test Files

```text
tests/
├── conftest.py
├── test_all_datasets.py
├── test_augmentations.py
├── test_datamodules.py
├── test_generic_image_dataset.py
└── test_datasets.py
```

`test_datamodules.py` covers the dataset factory and dataloader path.

## Running Tests

Run everything:

```bash
pytest
```

Verbose output:

```bash
pytest -v
```

Run one file:

```bash
pytest tests/test_datamodules.py
```

## Markers

Current markers used in the suite include:

- `unit`
- `integration`
- `requires_data`
- `slow`
- `augmentation`
- `datamodule`

Examples:

```bash
pytest -m unit
pytest -m "integration and requires_data"
pytest -m "not slow"
pytest -m augmentation
```

## Dataset-Dependent Tests

Some tests require real dataset files to be present locally. Those are marked
with:

- `integration`
- `requires_data`

If the expected dataset path is missing, those tests usually skip rather than fail.

## Common Usage Patterns

### Check Dataset Factory Wiring

```bash
pytest tests/test_datamodules.py -v
```

### Check Augmentation Construction

```bash
pytest tests/test_augmentations.py -v
```

### Run Only Fast Tests

```bash
pytest -m "unit and not slow"
```

## When Adding A New Dataset

When you add a new dataset, check at least:

1. the dataset config class exists
2. the dataset is registered in the dataset factory
3. one dataloader batch can be constructed

In practice, that usually means updating:

- `tests/conftest.py`
- and re-running:
  - `tests/test_datasets.py`
  - `tests/test_datamodules.py`

## When Adding A New Augmentation Policy

For a new augmentation policy, re-run:

```bash
pytest tests/test_augmentations.py -v
```

This is the quickest sanity check that the policy module still builds valid
train/test transforms.
