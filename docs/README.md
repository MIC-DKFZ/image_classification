# Documentation

This directory contains supplemental documentation for the current tyro +
pydantic + accelerate runtime.

Primary entrypoints:

- [README.md](../README.md): current top-level usage and architecture
- [src/glovita/configs](../src/glovita/configs): user-facing config schema
- [src/glovita/datasets/factory.py](../src/glovita/datasets/factory.py): dataset and dataloader assembly
- [src/glovita/augmentation/policies/registry.py](../src/glovita/augmentation/policies/registry.py): augmentation resolution

Useful docs in this folder:

- [augmentation/policies.md](augmentation/policies.md): augmentation layout and policy selection
- [datasets/DATASET_STRUCTURE.md](datasets/DATASET_STRUCTURE.md): dataset file layout expectations
- [mil.md](mil.md): MIL / CLAM / bag-of-features path
- [video.md](video.md): video encoder + framewise decoder structure
- [precomputed_features.md](precomputed_features.md): precomputed-feature loading and extraction
- [testing/TESTING.md](testing/TESTING.md): testing guide

Docs that still mention the old Hydra/Lightning stack should be treated as historical unless they explicitly describe the current `train.py` / `infer.py` runtime.
