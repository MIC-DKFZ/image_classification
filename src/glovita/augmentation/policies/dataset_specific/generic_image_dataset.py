from __future__ import annotations

from glovita.augmentation.policies.metadata import TrainPolicySpec


SPATIAL_DIM = 2

# The generic image dataset reuses the shared 2D policies directly. This module
# exists only so the registry can resolve the dataset name cleanly.
TRAIN_POLICIES: dict[str, TrainPolicySpec] = {}
TEST_POLICIES: dict[str, object] = {}
