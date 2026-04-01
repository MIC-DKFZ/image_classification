from __future__ import annotations

import timm
import torch
import torch.nn as nn


class TimmEncoder(nn.Module):
    def __init__(self, type: str, pretrained: bool, input_channels: int):
        super().__init__()
        self.model = timm.create_model(
            type,
            pretrained=pretrained,
            in_chans=input_channels,
            num_classes=0,
            global_pool="",
        )
        self.output_dim = int(self.model.num_features)
        self.features_are_tokens = bool(
            hasattr(self.model, "num_prefix_tokens")
            or hasattr(self.model, "patch_embed")
        )

    def forward_features(self, x):
        features = self.model.forward_features(x)
        if isinstance(features, (list, tuple)):
            features = features[0]
        if isinstance(features, dict):
            for key in ("x", "features", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Unsupported timm feature output type: {type(features)!r}")
        if features.ndim == 3:
            self.features_are_tokens = True
            return features
        if features.ndim == 4:
            self.features_are_tokens = False
            return features.mean(dim=(-2, -1))
        if features.ndim == 2:
            self.features_are_tokens = False
            return features
        raise ValueError(f"Unsupported timm feature shape: {tuple(features.shape)}")
