from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, JsonValue


AggregationMethod = Literal["cls_token", "avg", "sum", "mean_all", "joint"]


class TimmEncoderConfig(BaseModel):
    encoder_type: Literal["timm"] = "timm"
    type: str = "vit_base_patch16_224"
    pretrained: bool = True
    input_channels: int = 3
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict)


class TransformerEncoderConfig(BaseModel):
    encoder_type: Literal["transformer"] = "transformer"
    type: str = "facebook/vit-mae-base"
    pretrained: bool = True
    input_channels: int = 3
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict)


class TorchvisionEncoderConfig(BaseModel):
    encoder_type: Literal["torchvision"] = "torchvision"
    type: str = "convnext_tiny"
    pretrained: bool = True
    input_channels: int = 3
    dropout: float | None = Field(default=None, ge=0.0, lt=1.0)
    stochastic_depth_prob: float | None = Field(default=None, ge=0.0, lt=1.0)
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict)


class Dinov2EncoderConfig(BaseModel):
    encoder_type: Literal["dinov2"] = "dinov2"
    type: str = "dinov2_vitb14"
    input_channels: int = 3


class Dinov3EncoderConfig(BaseModel):
    encoder_type: Literal["dinov3"] = "dinov3"
    type: str = "dinov3_vitb16"
    weight_dir: Path
    input_channels: int = 3


class ResidualEncoderConfig(BaseModel):
    encoder_type: Literal["residual_encoder"] = "residual_encoder"
    input_channels: int = 3
    spatial_dim: Literal[2, 3] = 3
    features_per_stage: list[int] = Field(
        default_factory=lambda: [32, 64, 128, 256, 320, 320]
    )
    kernel_sizes: list[int] = Field(default_factory=lambda: [3, 3, 3, 3, 3, 3])
    strides: list[int] = Field(default_factory=lambda: [1, 2, 2, 2, 2, 2])
    n_blocks_per_stage: list[int] = Field(default_factory=lambda: [1, 3, 4, 6, 6, 6])
    conv_bias: bool = True
    norm: Literal["instance", "batch"] = "instance"
    dropout_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    nonlinearity: Literal["relu", "leaky_relu", "gelu"] = "leaky_relu"
    block_type: Literal["basic", "bottleneck"] = "basic"
    bottleneck_channels: list[int | None] | None = None
    stem_channels: int | None = None
    pool_type: Literal["conv", "avg", "max"] = "conv"
    stochastic_depth_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    squeeze_excitation: bool = False
    squeeze_excitation_reduction_ratio: float = Field(default=1 / 16, gt=0.0)


class PrimusEncoderConfig(BaseModel):
    encoder_type: Literal["primus"] = "primus"
    variant: Literal["S", "B", "M", "L"] = "S"
    input_channels: int = 3
    input_shape: tuple[int, int, int]
    patch_embed_size: tuple[int, int, int] = (8, 8, 8)
    drop_path_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    patch_drop_rate: float = Field(default=0.0, ge=0.0, lt=1.0)


class PrecomputedEncoderConfig(BaseModel):
    encoder_type: Literal["precomputed"] = "precomputed"
    feature_dim: int = Field(ge=1)


EncoderConfig = Annotated[
    Union[
        TimmEncoderConfig,
        TransformerEncoderConfig,
        TorchvisionEncoderConfig,
        Dinov2EncoderConfig,
        Dinov3EncoderConfig,
        ResidualEncoderConfig,
        PrimusEncoderConfig,
        PrecomputedEncoderConfig,
    ],
    Field(discriminator="encoder_type"),
]


class ClassificationHeadConfig(BaseModel):
    head_type: Literal["classification"] = "classification"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


class RegressionHeadConfig(BaseModel):
    head_type: Literal["regression"] = "regression"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    out_dim: int = Field(default=1, ge=1)


HeadConfig = Annotated[
    Union[ClassificationHeadConfig, RegressionHeadConfig],
    Field(discriminator="head_type"),
]


class ModelConfig(BaseModel):
    encoder: EncoderConfig = Field(default_factory=TimmEncoderConfig)
    head: HeadConfig = Field(default_factory=ClassificationHeadConfig)
    feature_aggregation_method: AggregationMethod = "cls_token"
