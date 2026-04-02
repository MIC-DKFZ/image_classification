from __future__ import annotations

import torch.nn as nn

from models.encoder.dinov2 import Dinov2Encoder
from models.encoder.dinov3 import Dinov3Encoder
from models.encoder.dynamic import PrimusEncoder, ResidualEncoder
from models.encoder.precomputed import PrecomputedEncoder
from models.encoder.timm import TimmEncoder
from models.encoder.torchvision import TorchvisionEncoder
from models.encoder.transformer import TransformerEncoder
from models.feature_aggregator import aggregate_features, aggregated_feature_dim
from models.heads.classification import ClassificationHead
from models.heads.regression import RegressionHead
from src.configs.model import (
    ClassificationHeadConfig,
    Dinov2EncoderConfig,
    Dinov3EncoderConfig,
    ModelConfig,
    PrecomputedEncoderConfig,
    PrimusEncoderConfig,
    RegressionHeadConfig,
    ResidualEncoderConfig,
    TimmEncoderConfig,
    TorchvisionEncoderConfig,
    TransformerEncoderConfig,
)


class ComposedModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, feature_aggregation_method: str):
        super().__init__()
        self.encoder = encoder
        # Keep .model and .cls_head for compatibility with existing PEFT utilities.
        self.model = encoder.model
        self.cls_head = head
        self.head = head
        self.feature_aggregation_method = feature_aggregation_method

    def _sync_encoder_model(self):
        self.encoder.model = self.model

    def extract_features(self, x):
        self._sync_encoder_model()
        return self.encoder.forward_features(x)

    def forward(self, x):
        features = self.extract_features(x)
        pooled = aggregate_features(features, self.feature_aggregation_method)
        return self.cls_head(pooled)


def build_encoder(config) -> nn.Module:
    if isinstance(config, TimmEncoderConfig):
        return TimmEncoder(
            type=config.type,
            pretrained=config.pretrained,
            input_channels=config.input_channels,
        )
    if isinstance(config, TransformerEncoderConfig):
        return TransformerEncoder(
            type=config.type,
            pretrained=config.pretrained,
            input_channels=config.input_channels,
        )
    if isinstance(config, TorchvisionEncoderConfig):
        return TorchvisionEncoder(
            type=config.type,
            pretrained=config.pretrained,
            input_channels=config.input_channels,
            dropout=config.dropout,
            stochastic_depth_prob=config.stochastic_depth_prob,
        )
    if isinstance(config, Dinov2EncoderConfig):
        return Dinov2Encoder(
            type=config.type,
        )
    if isinstance(config, Dinov3EncoderConfig):
        return Dinov3Encoder(
            type=config.type,
            weight_dir=config.weight_dir,
        )
    if isinstance(config, ResidualEncoderConfig):
        return ResidualEncoder(
            input_channels=config.input_channels,
            spatial_dim=config.spatial_dim,
            features_per_stage=config.features_per_stage,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            n_blocks_per_stage=config.n_blocks_per_stage,
            conv_bias=config.conv_bias,
            norm=config.norm,
            dropout_p=config.dropout_p,
            nonlinearity=config.nonlinearity,
            block_type=config.block_type,
            bottleneck_channels=config.bottleneck_channels,
            stem_channels=config.stem_channels,
            pool_type=config.pool_type,
            stochastic_depth_p=config.stochastic_depth_p,
            squeeze_excitation=config.squeeze_excitation,
            squeeze_excitation_reduction_ratio=config.squeeze_excitation_reduction_ratio,
        )
    if isinstance(config, PrimusEncoderConfig):
        return PrimusEncoder(
            variant=config.variant,
            input_channels=config.input_channels,
            input_shape=config.input_shape,
            patch_embed_size=config.patch_embed_size,
            drop_path_rate=config.drop_path_rate,
            patch_drop_rate=config.patch_drop_rate,
        )
    if isinstance(config, PrecomputedEncoderConfig):
        return PrecomputedEncoder(feature_dim=config.feature_dim)
    raise ValueError(f"Unsupported encoder config: {type(config).__name__}")


def build_head(config, input_dim: int, output_dim: int) -> nn.Module:
    if isinstance(config, ClassificationHeadConfig):
        return ClassificationHead(input_dim=input_dim, num_classes=output_dim, dropout=config.dropout)
    if isinstance(config, RegressionHeadConfig):
        return RegressionHead(input_dim=input_dim, out_dim=config.out_dim, dropout=config.dropout)
    raise ValueError(f"Unsupported head config: {type(config).__name__}")


def build_model(config: ModelConfig, output_dim: int) -> ComposedModel:
    encoder = build_encoder(config.encoder)
    pooled_dim = aggregated_feature_dim(
        embed_dim=encoder.output_dim,
        method=config.feature_aggregation_method,
        features_are_tokens=encoder.features_are_tokens,
    )
    head = build_head(config.head, pooled_dim, output_dim)
    return ComposedModel(
        encoder=encoder,
        head=head,
        feature_aggregation_method=config.feature_aggregation_method,
    )
