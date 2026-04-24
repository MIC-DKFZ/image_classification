# Documentation Index

This directory contains the more detailed documentation that sits underneath the
top-level [README.md](../README.md).

Start with the top-level README if you are new to the project. It explains:

- the config tree
- the CLI style
- run directory layout
- logger selection
- the extension path for datasets and models

Then use the topic-specific docs below.

## Core Docs

- [../README.md](../README.md): main project overview, CLI, config model, extension points
- [datasets/DATASET_STRUCTURE.md](datasets/DATASET_STRUCTURE.md): current split-file expectations and dataset layout rules
- [augmentation/policies.md](augmentation/policies.md): augmentation policy structure, available shared policies, and extension path
- [precomputed_features.md](precomputed_features.md): HDF5 feature training and extraction
- [mil.md](mil.md): MIL and CLAM over precomputed bags
- [video.md](video.md): video encoders, intermediates, and framewise heads
- [testing/TEST_RUN_README.md](testing/TEST_RUN_README.md): simple smoke-test examples
- [../tests/README.md](../tests/README.md): test suite structure and how to run it

## How To Use This Documentation

If you want to:

- run a normal experiment:
  - start with [../README.md](../README.md)
- understand dataset structure or add a dataset:
  - read [datasets/DATASET_STRUCTURE.md](datasets/DATASET_STRUCTURE.md)
- change augmentation defaults or add new policies:
  - read [augmentation/policies.md](augmentation/policies.md)
- train from extracted features:
  - read [precomputed_features.md](precomputed_features.md)
- use CLAM / bag-of-features MIL:
  - read [mil.md](mil.md)
- use video encoders or framewise heads:
  - read [video.md](video.md)
