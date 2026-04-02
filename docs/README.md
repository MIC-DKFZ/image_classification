# Documentation

This directory contains supplemental documentation for the current tyro +
pydantic + accelerate runtime.

Primary entrypoints:

- [README.md](/home/s522r/Desktop/classification_downstream/README.md): current top-level usage and architecture
- [src/configs](/home/s522r/Desktop/classification_downstream/src/configs): user-facing config schema
- [datasets/factory.py](/home/s522r/Desktop/classification_downstream/datasets/factory.py): dataset and dataloader assembly
- [augmentation/policies/registry.py](/home/s522r/Desktop/classification_downstream/augmentation/policies/registry.py): augmentation resolution

Useful docs in this folder:

- [augmentation/policies.md](/home/s522r/Desktop/classification_downstream/docs/augmentation/policies.md): augmentation layout and policy selection
- [datasets/DATASET_STRUCTURE.md](/home/s522r/Desktop/classification_downstream/docs/datasets/DATASET_STRUCTURE.md): dataset file layout expectations
- [testing/TESTING.md](/home/s522r/Desktop/classification_downstream/docs/testing/TESTING.md): testing guide

Docs that still mention the old Hydra/Lightning stack should be treated as historical unless they explicitly describe the current `train.py` / `infer.py` runtime.
