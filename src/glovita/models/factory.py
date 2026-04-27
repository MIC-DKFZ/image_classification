from __future__ import annotations

import torch.nn as nn

from glovita.models.img_encoder.dinov2 import Dinov2Encoder
from glovita.models.img_encoder.dinov3 import Dinov3Encoder
from glovita.models.img_encoder.dynamic import PrimusEncoder, ResidualEncoder
from glovita.models.img_encoder.precomputed import PrecomputedEncoder
from glovita.models.img_encoder.timm import TimmEncoder
from glovita.models.img_encoder.torchvision import TorchvisionEncoder
from glovita.models.img_encoder.transformer import TransformerEncoder
from glovita.models.feature_aggregator import aggregate_features, aggregated_feature_dim
from glovita.models.heads.classification import ClassificationHead
from glovita.models.heads.mil.clam import CLAM_MB, CLAM_SB
from glovita.models.heads.regression import RegressionHead
from glovita.models.heads.video.framewise_decoder_1d import FramewiseDecoder1D
from glovita.models.video_encoder.pytorchvideo import PytorchvideoEncoder
from glovita.models.video_encoder.torchvision import TorchvisionVideoEncoder
from glovita.configs.model import (
    ClamHeadConfig,
    ClassificationHeadConfig,
    Dinov2EncoderConfig,
    Dinov3EncoderConfig,
    FramewiseDecoder1DHeadConfig,
    ModelConfig,
    PytorchvideoEncoderConfig,
    PrecomputedEncoderConfig,
    PrimusEncoderConfig,
    RegressionHeadConfig,
    ResidualEncoderConfig,
    TimmEncoderConfig,
    TorchvisionVideoEncoderConfig,
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
        self.latest_aux_loss = None

    def _sync_encoder_model(self):
        self.encoder.model = self.model

    def extract_features(self, x):
        self._sync_encoder_model()
        return self.encoder.forward_features(x)

    def forward(self, x, target=None):
        features = self.extract_features(x)
        self.latest_aux_loss = None

        if getattr(self.cls_head, "consumes_raw_features", False):
            logits = self.cls_head(features, label=target)
            if hasattr(self.cls_head, "get_aux_loss"):
                self.latest_aux_loss = self.cls_head.get_aux_loss()
            return logits

        if isinstance(features, dict):
            if "mask" in features:
                raise ValueError(
                    "Bag-style feature inputs require a head that consumes raw bag features, such as `clam`."
                )
            tensor_features = features.get("features")
            if tensor_features is None:
                raise ValueError(
                    "Structured encoder outputs for pooled heads must include a `features` entry."
                )
        else:
            tensor_features = features
        pooled = aggregate_features(tensor_features, self.feature_aggregation_method)
        return self.cls_head(pooled)


def build_encoder(config) -> nn.Module:
    if isinstance(config, TimmEncoderConfig):
        return TimmEncoder(
            type=config.type,
            pretrained=config.pretrained,
            input_channels=config.input_channels,
            model_kwargs=config.model_kwargs,
        )
    if isinstance(config, TransformerEncoderConfig):
        return TransformerEncoder(
            type=config.type,
            pretrained=config.pretrained,
            input_channels=config.input_channels,
            model_kwargs=config.model_kwargs,
        )
    if isinstance(config, TorchvisionEncoderConfig):
        return TorchvisionEncoder(
            type=config.type,
            pretrained=config.pretrained,
            input_channels=config.input_channels,
            dropout=config.dropout,
            stochastic_depth_prob=config.stochastic_depth_prob,
            model_kwargs=config.model_kwargs,
        )
    if isinstance(config, TorchvisionVideoEncoderConfig):
        return TorchvisionVideoEncoder(
            type=config.type,
            pretrained=config.pretrained,
            input_channels=config.input_channels,
            return_intermediates=config.return_intermediates,
            intermediate_names=config.intermediate_names,
            model_kwargs=config.model_kwargs,
        )
    if isinstance(config, PytorchvideoEncoderConfig):
        return PytorchvideoEncoder(
            type=config.type,
            pretrained=config.pretrained,
            input_channels=config.input_channels,
            pathway_mode=config.pathway_mode,
            slowfast_alpha=config.slowfast_alpha,
            return_intermediates=config.return_intermediates,
            intermediate_names=config.intermediate_names,
            model_kwargs=config.model_kwargs,
        )
    if isinstance(config, Dinov2EncoderConfig):
        return Dinov2Encoder(
            type=config.type,
            pretrained=config.pretrained,
            weight_dir=config.weight_dir,
        )
    if isinstance(config, Dinov3EncoderConfig):
        return Dinov3Encoder(
            type=config.type,
            pretrained=config.pretrained,
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
    if isinstance(config, ClamHeadConfig):
        head_cls = CLAM_SB if config.variant == "sb" else CLAM_MB
        return head_cls(
            gate=config.gate,
            size_arg=config.size_arg,
            dropout=config.dropout,
            k_sample=config.k_sample,
            n_classes=output_dim,
            subtyping=config.subtyping,
            embed_dim=input_dim,
            feature_prep=config.feature_prep,
            l2_normalize_features=config.l2_normalize_features,
            layer_norm_eps=config.layer_norm_eps,
            cosine_head=config.cosine_head,
            cosine_scale=config.cosine_scale,
            instance_eval=config.instance_eval,
            instance_loss_weight=config.instance_loss_weight,
            attn_drop=config.attn_drop,
            topk_k=config.topk_k,
            topk_tau=config.topk_tau,
            stochastic_topk=config.stochastic_topk,
            topk_noise_std=config.topk_noise_std,
            topk_consistency_weight=config.topk_consistency_weight,
        )
    if isinstance(config, FramewiseDecoder1DHeadConfig):
        return FramewiseDecoder1D(
            num_classes=output_dim,
            num_clip_frames=config.num_clip_frames,
            stem_key=config.stem_key,
            layer2_key=config.layer2_key,
            layer3_key=config.layer3_key,
            layer4_key=config.layer4_key,
        )
    if isinstance(config, RegressionHeadConfig):
        return RegressionHead(input_dim=input_dim, out_dim=config.out_dim, dropout=config.dropout)
    raise ValueError(f"Unsupported head config: {type(config).__name__}")


def build_model(config: ModelConfig, output_dim: int) -> ComposedModel:
    encoder = build_encoder(config.encoder)
    head_uses_raw_features = isinstance(config.head, (ClamHeadConfig, FramewiseDecoder1DHeadConfig))
    if head_uses_raw_features:
        head_input_dim = encoder.output_dim
    else:
        head_input_dim = aggregated_feature_dim(
            embed_dim=encoder.output_dim,
            method=config.feature_aggregation_method,
            features_are_tokens=encoder.features_are_tokens,
        )
    head = build_head(config.head, head_input_dim, output_dim)
    return ComposedModel(
        encoder=encoder,
        head=head,
        feature_aggregation_method=config.feature_aggregation_method,
    )
