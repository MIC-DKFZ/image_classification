from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class Dinov3Encoder(nn.Module):
    def __init__(self, type: str, pretrained: bool = True, weight_dir: Path | None = None):
        super().__init__()
        model_name = "_".join(type.split("_")[:2])
        self.model = torch.hub.load("facebookresearch/dinov3", model_name, pretrained=pretrained)
        if weight_dir is not None:
            state_dict = torch.load(Path(weight_dir) / f"{type}.pth", map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
        self.output_dim = int(self.model.embed_dim)
        self.features_are_tokens = True

    def forward_features(self, x):
        features = self.model.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        return torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
