from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, JsonValue


AggregationMethod = Literal["cls_token", "avg", "sum", "mean_all", "joint"]


class TimmEncoderConfig(BaseModel):
    """Image encoder from the `timm` model zoo."""

    encoder_type: Literal["timm"] = "timm"
    type: str = Field(default="vit_base_patch16_224", description="timm model name passed to timm.create_model.")
    pretrained: bool = Field(default=True, description="Load pretrained weights from timm when available.")
    input_channels: int = Field(default=3, description="Number of input image channels.")
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict, description="Extra timm.create_model kwargs for architecture-specific settings.")


class TransformerEncoderConfig(BaseModel):
    """Image encoder backed by Hugging Face Transformers."""

    encoder_type: Literal["transformer"] = "transformer"
    type: str = Field(default="facebook/vit-mae-base", description="Transformers model identifier.")
    pretrained: bool = Field(default=True, description="Load pretrained weights from Hugging Face.")
    input_channels: int = Field(default=3, description="Number of input image channels.")
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict, description="Extra model-construction kwargs for this encoder family.")


class TorchvisionEncoderConfig(BaseModel):
    """Image encoder from `torchvision.models`."""

    encoder_type: Literal["torchvision"] = "torchvision"
    type: str = Field(default="convnext_tiny", description="torchvision model name.")
    pretrained: bool = Field(default=True, description="Load pretrained torchvision weights when available.")
    input_channels: int = Field(default=3, description="Number of input image channels.")
    dropout: float | None = Field(default=None, ge=0.0, lt=1.0, description="Optional classifier/head dropout passed to supported torchvision models.")
    stochastic_depth_prob: float | None = Field(default=None, ge=0.0, lt=1.0, description="Optional stochastic depth probability for supported models.")
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict, description="Extra torchvision model kwargs for architecture-specific settings.")


class TorchvisionVideoEncoderConfig(BaseModel):
    """Clip-level video encoder from `torchvision.models.video`."""

    encoder_type: Literal["torchvision_video"] = "torchvision_video"
    type: str = Field(default="r3d_18", description="torchvision video model name.")
    pretrained: bool = Field(default=True, description="Load pretrained torchvision video weights when available.")
    input_channels: int = Field(default=3, description="Number of input channels per frame.")
    return_intermediates: bool = Field(default=False, description="Return named intermediate feature stages in addition to the final features.")
    intermediate_names: list[str] = Field(default_factory=list, description="Requested intermediate stage names. Leave empty to use encoder defaults when available.")
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict, description="Extra model kwargs for this video encoder.")


class PytorchvideoEncoderConfig(BaseModel):
    """Clip-level video encoder from `pytorchvideo`."""

    encoder_type: Literal["pytorchvideo"] = "pytorchvideo"
    type: str = Field(default="x3d_s", description="pytorchvideo model name.")
    pretrained: bool = Field(default=True, description="Load pretrained pytorchvideo weights when available.")
    input_channels: int = Field(default=3, description="Number of input channels per frame.")
    pathway_mode: Literal["auto", "single", "slowfast"] = Field(default="auto", description="How to prepare pathway inputs for single-pathway versus SlowFast-style models.")
    slowfast_alpha: int = Field(default=4, ge=1, description="Temporal stride ratio between slow and fast pathways when using SlowFast inputs.")
    return_intermediates: bool = Field(default=False, description="Return named intermediate feature stages in addition to the final features.")
    intermediate_names: list[str] = Field(default_factory=list, description="Requested intermediate stage names. Leave empty to use encoder defaults when available.")
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict, description="Extra model kwargs for this video encoder.")


class Dinov2EncoderConfig(BaseModel):
    """DINOv2 image encoder loaded through torch.hub."""

    encoder_type: Literal["dinov2"] = "dinov2"
    type: str = Field(default="dinov2_vitb14", description="DINOv2 model identifier.")
    pretrained: bool = Field(default=True, description="Load the default DINOv2 pretrained weights.")
    weight_dir: Path | None = Field(default=None, description="Optional local directory containing <type>.pth to override the default weights.")
    input_channels: int = Field(default=3, description="Number of input image channels.")


class Dinov3EncoderConfig(BaseModel):
    """DINOv3 image encoder loaded through torch.hub."""

    encoder_type: Literal["dinov3"] = "dinov3"
    type: str = Field(default="dinov3_vitb16", description="DINOv3 model identifier.")
    pretrained: bool = Field(default=True, description="Load the default DINOv3 pretrained weights.")
    weight_dir: Path | None = Field(default=None, description="Optional local directory containing <type>.pth to override the default weights.")
    input_channels: int = Field(default=3, description="Number of input image channels.")


class ResidualEncoderConfig(BaseModel):
    """Residual encoder for volumetric or image inputs built from dynamic-network-architectures."""

    encoder_type: Literal["residual_encoder"] = "residual_encoder"
    input_channels: int = Field(default=3, description="Number of input channels.")
    spatial_dim: Literal[2, 3] = Field(default=3, description="Spatial dimensionality of the input tensor.")
    features_per_stage: list[int] = Field(
        default_factory=lambda: [32, 64, 128, 256, 320, 320],
        description="Channel width for each encoder stage.",
    )
    kernel_sizes: list[int] = Field(default_factory=lambda: [3, 3, 3, 3, 3, 3], description="Kernel size for each encoder stage.")
    strides: list[int] = Field(default_factory=lambda: [1, 2, 2, 2, 2, 2], description="Stride for each encoder stage.")
    n_blocks_per_stage: list[int] = Field(default_factory=lambda: [1, 3, 4, 6, 6, 6], description="Number of residual blocks in each stage.")
    conv_bias: bool = Field(default=True, description="Use bias parameters in convolution layers.")
    norm: Literal["instance", "batch"] = Field(default="instance", description="Normalization layer type.")
    dropout_p: float = Field(default=0.0, ge=0.0, lt=1.0, description="Dropout probability inside residual blocks.")
    nonlinearity: Literal["relu", "leaky_relu", "gelu"] = Field(default="leaky_relu", description="Activation function.")
    block_type: Literal["basic", "bottleneck"] = Field(default="basic", description="Residual block variant.")
    bottleneck_channels: list[int | None] | None = Field(default=None, description="Optional bottleneck width per stage.")
    stem_channels: int | None = Field(default=None, description="Optional explicit width of the stem convolution.")
    pool_type: Literal["conv", "avg", "max"] = Field(default="conv", description="Downsampling operator type.")
    stochastic_depth_p: float = Field(default=0.0, ge=0.0, lt=1.0, description="Stochastic depth probability.")
    squeeze_excitation: bool = Field(default=False, description="Enable squeeze-excitation blocks.")
    squeeze_excitation_reduction_ratio: float = Field(default=1 / 16, gt=0.0, description="Squeeze-excitation reduction ratio.")


class PrimusEncoderConfig(BaseModel):
    """PRIMUS encoder for 3D inputs."""

    encoder_type: Literal["primus"] = "primus"
    variant: Literal["S", "B", "M", "L"] = Field(default="S", description="PRIMUS model scale.")
    input_channels: int = Field(default=3, description="Number of input channels.")
    input_shape: tuple[int, int, int] = Field(description="Expected 3D input shape.")
    patch_embed_size: tuple[int, int, int] = Field(default=(8, 8, 8), description="Patch embedding size.")
    drop_path_rate: float = Field(default=0.0, ge=0.0, lt=1.0, description="Stochastic depth rate.")
    patch_drop_rate: float = Field(default=0.0, ge=0.0, lt=1.0, description="Patch dropout rate.")


class PrecomputedEncoderConfig(BaseModel):
    """Identity encoder for precomputed feature files."""

    encoder_type: Literal["precomputed"] = "precomputed"
    feature_dim: int = Field(ge=1, description="Feature dimension stored in the input HDF5 files.")


EncoderConfig = Annotated[
    Union[
        TimmEncoderConfig,
        TransformerEncoderConfig,
        TorchvisionEncoderConfig,
        TorchvisionVideoEncoderConfig,
        PytorchvideoEncoderConfig,
        Dinov2EncoderConfig,
        Dinov3EncoderConfig,
        ResidualEncoderConfig,
        PrimusEncoderConfig,
        PrecomputedEncoderConfig,
    ],
    Field(discriminator="encoder_type"),
]


class ClassificationHeadConfig(BaseModel):
    """Standard classification head over pooled encoder features."""

    head_type: Literal["classification"] = "classification"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0, description="Dropout applied before the classifier.")


class ClamHeadConfig(BaseModel):
    """CLAM multiple-instance learning head over bags of features."""

    head_type: Literal["clam"] = "clam"
    variant: Literal["sb", "mb"] = Field(default="mb", description="Single-branch or multi-branch CLAM variant.")
    gate: bool = Field(default=True, description="Use gated attention.")
    size_arg: Literal["tiny", "small", "big"] = Field(default="small", description="Hidden-size preset for the CLAM MLPs.")
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0, description="Dropout probability in the CLAM head.")
    k_sample: int = Field(default=8, ge=1, description="Number of top positive/negative instances used by instance-level supervision.")
    subtyping: bool = Field(default=False, description="Enable out-of-class instance supervision.")
    feature_prep: bool = Field(default=True, description="Apply the CLAM feature preprocessing block.")
    l2_normalize_features: bool = Field(default=True, description="L2-normalize features inside the feature preprocessing block.")
    layer_norm_eps: float = Field(default=1e-6, gt=0.0, description="LayerNorm epsilon in the CLAM feature preprocessing block.")
    cosine_head: bool = Field(default=True, description="Use cosine classifiers instead of standard linear heads.")
    cosine_scale: float = Field(default=20.0, gt=0.0, description="Scale factor for cosine classifier logits.")
    instance_eval: bool = Field(default=False, description="Enable the auxiliary CLAM instance-level loss.")
    instance_loss_weight: float = Field(default=1.0, ge=0.0, description="Weight of the auxiliary instance-level loss.")
    attn_drop: float = Field(default=0.0, ge=0.0, lt=1.0, description="Dropout probability in the attention module.")
    topk_k: int = Field(default=2, ge=1, description="Number of top-attended instances used by the stochastic top-k branch.")
    topk_tau: float = Field(default=0.25, gt=0.0, description="Temperature used by the stochastic top-k branch.")
    stochastic_topk: bool = Field(default=False, description="Enable the auxiliary stochastic top-k consistency branch.")
    topk_noise_std: float = Field(default=0.1, ge=0.0, description="Gaussian noise std used to perturb top-k instances.")
    topk_consistency_weight: float = Field(default=1.0, ge=0.0, description="Weight of the stochastic top-k consistency penalty.")


class FramewiseDecoder1DHeadConfig(BaseModel):
    """Framewise decoder head that consumes named video encoder stages."""

    head_type: Literal["framewise_decoder_1d"] = "framewise_decoder_1d"
    num_clip_frames: int = Field(default=3, ge=1, description="Number of frames per clip expected by the decoder.")
    stem_key: str = Field(default="stem", description="Intermediate stage name used for stem features.")
    layer2_key: str = Field(default="layer2", description="Intermediate stage name used for layer2 features.")
    layer3_key: str = Field(default="layer3", description="Intermediate stage name used for layer3 features.")
    layer4_key: str = Field(default="layer4", description="Intermediate stage name used for layer4 features.")


class RegressionHeadConfig(BaseModel):
    """Standard regression head over pooled encoder features."""

    head_type: Literal["regression"] = "regression"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0, description="Dropout applied before the regressor.")
    out_dim: int = Field(default=1, ge=1, description="Number of regression outputs.")


HeadConfig = Annotated[
    Union[
        ClassificationHeadConfig,
        ClamHeadConfig,
        FramewiseDecoder1DHeadConfig,
        RegressionHeadConfig,
    ],
    Field(discriminator="head_type"),
]


class ModelConfig(BaseModel):
    """Top-level model composition: encoder, head, and feature aggregation."""

    encoder: EncoderConfig = Field(default_factory=TimmEncoderConfig, description="Encoder family and encoder-specific parameters.")
    head: HeadConfig = Field(default_factory=ClassificationHeadConfig, description="Prediction head configuration.")
    feature_aggregation_method: AggregationMethod = Field(
        default="cls_token",
        description="How token-like encoder features are pooled before the head: cls_token = CLS token only, avg = mean of patch tokens only, sum = sum of patch tokens, mean_all = mean over all tokens including CLS, joint = concatenate CLS token with mean patch token.",
    )
