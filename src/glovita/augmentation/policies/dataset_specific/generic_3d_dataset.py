from __future__ import annotations

from glovita.augmentation.policies.metadata import TrainPolicySpec


SPATIAL_DIM = 3

# The generic 3D dataset reuses the shared 3D policies directly. This module
# exists only so the registry can resolve the dataset name cleanly.
TRAIN_POLICIES: dict[str, TrainPolicySpec] = {}
TEST_POLICIES: dict[str, object] = {}
