from __future__ import annotations

import torch.nn as nn


class PrecomputedEncoder(nn.Module):
    """Identity encoder for precomputed feature tensors."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.model = nn.Identity()
        self.output_dim = int(feature_dim)
        self.features_are_tokens = False

    def forward_features(self, x):
        return x
