from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


_PRIMUS_EMBED_DIMS = {
    "S": 396,
    "B": 792,
    "M": 864,
    "L": 1056,
}


def _get_conv_op(spatial_dim: int):
    if spatial_dim == 2:
        return nn.Conv2d
    if spatial_dim == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported spatial_dim={spatial_dim}. Expected 2 or 3.")


def _get_norm_op(spatial_dim: int, norm: Literal["instance", "batch"]):
    if norm == "instance":
        return nn.InstanceNorm2d if spatial_dim == 2 else nn.InstanceNorm3d
    if norm == "batch":
        return nn.BatchNorm2d if spatial_dim == 2 else nn.BatchNorm3d
    raise ValueError(f"Unsupported norm={norm!r}.")


def _get_nonlin(nonlin: Literal["relu", "leaky_relu", "gelu"]):
    if nonlin == "relu":
        return nn.ReLU, {}
    if nonlin == "leaky_relu":
        return nn.LeakyReLU, {"inplace": True}
    if nonlin == "gelu":
        return nn.GELU, {}
    raise ValueError(f"Unsupported nonlinearity={nonlin!r}.")


class ResidualEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        spatial_dim: int,
        features_per_stage: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        n_blocks_per_stage: list[int],
        conv_bias: bool,
        norm: Literal["instance", "batch"],
        dropout_p: float,
        nonlinearity: Literal["relu", "leaky_relu", "gelu"],
        block_type: Literal["basic", "bottleneck"],
        bottleneck_channels: list[int | None] | None,
        stem_channels: int | None,
        pool_type: Literal["conv", "avg", "max"],
        stochastic_depth_p: float,
        squeeze_excitation: bool,
        squeeze_excitation_reduction_ratio: float,
    ):
        super().__init__()
        from dynamic_network_architectures.building_blocks.residual import (
            BasicBlockD,
            BottleneckD,
        )
        from dynamic_network_architectures.building_blocks.residual_encoders import (
            ResidualEncoder as DynamicResidualEncoder,
        )

        if len(features_per_stage) != len(kernel_sizes):
            raise ValueError(
                "features_per_stage and kernel_sizes must have the same length."
            )
        if len(features_per_stage) != len(strides):
            raise ValueError(
                "features_per_stage and strides must have the same length."
            )
        if len(features_per_stage) != len(n_blocks_per_stage):
            raise ValueError(
                "features_per_stage and n_blocks_per_stage must have the same length."
            )

        conv_op = _get_conv_op(spatial_dim)
        norm_op = _get_norm_op(spatial_dim, norm)
        nonlin, nonlin_kwargs = _get_nonlin(nonlinearity)
        block = BasicBlockD if block_type == "basic" else BottleneckD
        dropout_op = nn.Dropout2d if spatial_dim == 2 else nn.Dropout3d

        self.model = DynamicResidualEncoder(
            input_channels=input_channels,
            n_stages=len(features_per_stage),
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=dropout_op if dropout_p > 0 else None,
            dropout_op_kwargs=(
                {"p": dropout_p, "inplace": False} if dropout_p > 0 else None
            ),
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=block,
            bottleneck_channels=bottleneck_channels,
            return_skips=False,
            stem_channels=stem_channels,
            pool_type=pool_type,
            stochastic_depth_p=stochastic_depth_p,
            squeeze_excitation=squeeze_excitation,
            squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
        )
        self.output_dim = int(features_per_stage[-1])
        self.features_are_tokens = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Residual encoder returned unsupported type: {type(features)!r}"
            )
        if features.ndim < 3:
            return features
        return features.mean(dim=tuple(range(2, features.ndim)))


class PrimusEncoder(nn.Module):
    def __init__(
        self,
        variant: Literal["S", "B", "M", "L"],
        input_channels: int,
        input_shape: tuple[int, int, int],
        patch_embed_size: tuple[int, int, int],
        drop_path_rate: float,
        patch_drop_rate: float,
    ):
        super().__init__()
        from dynamic_network_architectures.architectures.primus import (
            PrimusB,
            PrimusL,
            PrimusM,
            PrimusS,
        )

        constructors = {
            "S": PrimusS,
            "B": PrimusB,
            "M": PrimusM,
            "L": PrimusL,
        }
        self.model = constructors[variant](
            input_channels=input_channels,
            output_channels=1,
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
        )
        self.output_dim = _PRIMUS_EMBED_DIMS[variant]
        self.features_are_tokens = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.down_projection(x)
        x = x.flatten(2).transpose(1, 2)

        register_tokens = getattr(self.model, "register_tokens", None)
        if register_tokens is not None:
            x = torch.cat((register_tokens.expand(x.shape[0], -1, -1), x), dim=1)

        x, _ = self.model.eva(x)
        if register_tokens is not None:
            x = x[:, register_tokens.shape[1] :]

        return x.mean(dim=1)
