from __future__ import annotations

import torch
import torch.nn as nn


def _double_conv_1d(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


class FramewiseDecoder1D(nn.Module):
    """Temporal decoder for framewise video prediction from multi-scale 3D features.

    Expected input is a dict with feature maps from a ResNet-style 3D encoder:
    `stem`, `layer2`, `layer3`, `layer4`.
    """

    consumes_raw_features = True

    def __init__(
        self,
        num_classes: int,
        num_clip_frames: int = 3,
        stem_key: str = "stem",
        layer2_key: str = "layer2",
        layer3_key: str = "layer3",
        layer4_key: str = "layer4",
    ):
        super().__init__()
        self.stem_key = stem_key
        self.layer2_key = layer2_key
        self.layer3_key = layer3_key
        self.layer4_key = layer4_key
        num_stages = 3
        self.dconv_up3 = _double_conv_1d(512 + 256, 256)
        self.dconv_up2 = _double_conv_1d(256 + 128, 128)
        self.dconv_up1 = _double_conv_1d(128 + 64, 64)

        self.upsample3 = nn.Upsample(
            num_clip_frames // num_stages,
            mode="linear",
            align_corners=True,
        )
        self.upsample2 = nn.Upsample(
            num_clip_frames // (num_stages - 1),
            mode="linear",
            align_corners=True,
        )
        self.upsample1 = nn.Upsample(
            num_clip_frames // (num_stages - 2),
            mode="linear",
            align_corners=True,
        )

        self.conv_last = nn.Conv1d(64, num_classes, kernel_size=1)
        self.adaptive_pool3 = nn.AdaptiveAvgPool3d((num_clip_frames // num_stages, 1, 1))
        self.adaptive_pool2 = nn.AdaptiveAvgPool3d((num_clip_frames // (num_stages - 1), 1, 1))
        self.adaptive_pool1 = nn.AdaptiveAvgPool3d((num_clip_frames // (num_stages - 2), 1, 1))

    def forward(self, features: dict[str, torch.Tensor], label=None) -> torch.Tensor:
        _ = label
        if "intermediates" in features:
            features = features["intermediates"]
        x1 = features[self.stem_key]
        x3 = features[self.layer2_key]
        x4 = features[self.layer3_key]
        x5 = features[self.layer4_key]

        x = x5.mean(dim=(-2, -1), keepdim=True)
        x = self.upsample3(x.squeeze((-3, -2, -1)).unsqueeze(-1))
        x = self.dconv_up3(
            torch.cat([x, self.adaptive_pool3(x4).squeeze((-2, -1))], dim=1)
        )
        x = self.upsample2(x)
        x = self.dconv_up2(
            torch.cat([x, self.adaptive_pool2(x3).squeeze((-2, -1))], dim=1)
        )
        x = self.upsample1(x)
        x = self.dconv_up1(
            torch.cat([x, self.adaptive_pool1(x1).squeeze((-2, -1))], dim=1)
        )
        return self.conv_last(x)
