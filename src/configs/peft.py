from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Simple freeze / unfreeze methods
# ---------------------------------------------------------------------------

class LinearProbingConfig(BaseModel):
    """Train only the classification head; freeze the entire backbone."""
    method: Literal["linear_probing"] = "linear_probing"


class FullFinetuningConfig(BaseModel):
    """Train all parameters. Supports layer-wise LR decay via OptimizerConfig."""
    method: Literal["full_finetuning"] = "full_finetuning"


class PartialFinetuningConfig(BaseModel):
    """Freeze the first N transformer blocks; fine-tune the rest + head."""
    method: Literal["partial_finetuning"] = "partial_finetuning"
    num_frozen_layers: int = Field(default=6, ge=0)


class BitFitConfig(BaseModel):
    """BitFit (ACL 2022): train only bias terms and the classification head."""
    method: Literal["bitfit"] = "bitfit"


class LNTuningConfig(BaseModel):
    """Train only LayerNorm parameters and the classification head."""
    method: Literal["ln_tuning"] = "ln_tuning"


class SSFConfig(BaseModel):
    """SSF: per-channel scale & shift for every activation."""
    method: Literal["ssf"] = "ssf"


class DiffFitConfig(BaseModel):
    """DiffFit: bias terms + per-channel gamma scales."""
    method: Literal["difffit"] = "difffit"


# ---------------------------------------------------------------------------
# Low-rank decomposition methods (via HuggingFace PEFT library)
# ---------------------------------------------------------------------------

class LoraConfig(BaseModel):
    """LoRA / DoRA (ICLR 2022 / 2024): low-rank weight updates.

    Set use_dora=True to activate DoRA (weight decomposition LoRA).
    """
    method: Literal["lora"] = "lora"
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: float = Field(default=16.0, gt=0.0)
    lora_dropout: float = Field(default=0.01, ge=0.0, lt=1.0)
    lora_target_modules: List[str] = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.proj", "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2",
    ]
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: bool = True
    lora_bias: Literal["none", "all", "lora_only"] = "none"


class AdaLoraConfig(BaseModel):
    """AdaLoRA: adaptive rank allocation via singular value decomposition."""
    method: Literal["adalora"] = "adalora"
    adalora_rank: int = Field(default=4, ge=1)
    adalora_init_rank: int = Field(default=12, ge=1)
    adalora_alpha: float = Field(default=32.0, gt=0.0)
    adalora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    adalora_target_modules: List[str] = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.proj", "mlp.fc1", "mlp.fc2",
    ]
    use_rslora: bool = False
    init_lora_weights: bool = True
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    adalora_orth_reg_weight: float = 0.5
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_tinit: int = 0
    adalora_deltaT: int = 1


class LoHaConfig(BaseModel):
    """LoHa: Low-rank Hadamard product adaptation."""
    method: Literal["loha"] = "loha"
    loha_rank: int = Field(default=4, ge=1)
    loha_alpha: float = Field(default=1.0, gt=0.0)
    loha_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    loha_rank_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    loha_target_modules: List[str] = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.proj", "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2",
    ]


class LoKrConfig(BaseModel):
    """LoKr: Low-rank Kronecker product adaptation."""
    method: Literal["lokr"] = "lokr"
    lokr_rank: int = Field(default=4, ge=1)
    lokr_alpha: float = Field(default=1.0, gt=0.0)
    lokr_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    lokr_rank_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    lokr_decompose_factor: int = -1
    lokr_target_modules: List[str] = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.proj", "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2",
    ]


class OFTConfig(BaseModel):
    """OFT: Orthogonal Fine-Tuning via orthogonal matrices."""
    method: Literal["oft"] = "oft"
    oft_r: int = Field(default=8, ge=1)
    oft_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    oft_target_modules: List[str] = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.proj", "mlp.fc1", "mlp.fc2",
    ]
    oft_coft: bool = False
    oft_eps: float = 6e-5
    oft_bias: Literal["none", "all"] = "none"
    oft_use_cayley_neumann: bool = True


class BOFTConfig(BaseModel):
    """BOFT: Butterfly Orthogonal Fine-Tuning."""
    method: Literal["boft"] = "boft"
    boft_block_size: int = Field(default=8, ge=1)
    boft_n_butterfly_factor: int = Field(default=1, ge=1)
    boft_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    boft_target_modules: List[str] = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.proj", "mlp.fc1", "mlp.fc2",
    ]
    boft_bias: Literal["none", "all"] = "none"


class VeRAConfig(BaseModel):
    """VeRA: Very lightweight LoRA variant with shared random matrices."""
    method: Literal["vera"] = "vera"
    vera_rank: int = Field(default=8, ge=1)
    vera_dropout: float = Field(default=0.01, ge=0.0, lt=1.0)
    vera_target_modules: List[str] = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.proj", "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2",
    ]
    vera_projection_prng_key: int = 0


class IA3Config(BaseModel):
    """IA³: per-channel learned scaling vectors for attention and FFN."""
    method: Literal["ia3"] = "ia3"


class FourierFTConfig(BaseModel):
    """FourierFT: fine-tuning in the Fourier frequency domain."""
    method: Literal["fourierft"] = "fourierft"
    fourierft_n_frequency: int = Field(default=1000, ge=1)
    fourierft_scaling: float = Field(default=150.0, gt=0.0)
    fourierft_target_modules: List[str] = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.proj", "mlp.fc1", "mlp.fc2",
    ]


# ---------------------------------------------------------------------------
# Adapter-based methods
# ---------------------------------------------------------------------------

class AdaptFormerConfig(BaseModel):
    """AdaptFormer: lightweight MLP adapters injected in parallel with FFN."""
    method: Literal["adapt_former"] = "adapt_former"
    mode: Literal["parallel", "sequential"] = "parallel"
    bottleneck: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    init_option: Literal["lora", "bert"] = "lora"
    adapter_scalar: float = 1.0
    adapter_layernorm_option: Literal["in", "out", "none"] = "in"
    freeze_backbone: bool = True


class ConvpassConfig(BaseModel):
    """ConvPass: convolutional bypass adapter with spatial 2D processing."""
    method: Literal["convpass"] = "convpass"
    bottleneck: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    freeze_backbone: bool = True


class RepAdapterConfig(BaseModel):
    """RepAdapter: structurally re-parameterisable adapter."""
    method: Literal["rep_adapter"] = "rep_adapter"
    repadapter_bottleneck: int = Field(default=8, ge=1)
    repadapter_groups: int = Field(default=2, ge=1)
    repadapter_scale_init: float = 0.001


# ---------------------------------------------------------------------------
# Structured / specialist methods
# ---------------------------------------------------------------------------

class FacTConfig(BaseModel):
    """FacT: Factor-Tuning via Tucker decomposition."""
    method: Literal["fact"] = "fact"
    fact_r: int = Field(default=4, ge=1)


class GPSConfig(BaseModel):
    """GPS: Gradient-based Parameter Selection."""
    method: Literal["gps"] = "gps"
    gps_percent: float = Field(default=1.0, gt=0.0, le=100.0)


class VisualPromptTuningConfig(BaseModel):
    """VPT: Visual Prompt Tuning — prepend learnable tokens to patch sequences.

    Set deep=True for deep VPT (prompts at every transformer block).
    Set deep=False for shallow VPT (prompts only at the first block).
    """
    method: Literal["visual_prompt_tuning"] = "visual_prompt_tuning"
    num_tokens: int = Field(default=20, ge=1)
    deep: bool = True
    project_dim: Optional[int] = None
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    init_scale: float = 1.0
    # Which transformer blocks receive prompts (None = all). Only used when deep=True.
    deep_layers: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Discriminated union — the single type used everywhere
# ---------------------------------------------------------------------------

PeftConfig = Annotated[
    Union[
        LinearProbingConfig,
        FullFinetuningConfig,
        PartialFinetuningConfig,
        BitFitConfig,
        LNTuningConfig,
        SSFConfig,
        DiffFitConfig,
        LoraConfig,
        AdaLoraConfig,
        LoHaConfig,
        LoKrConfig,
        OFTConfig,
        BOFTConfig,
        VeRAConfig,
        IA3Config,
        FourierFTConfig,
        AdaptFormerConfig,
        ConvpassConfig,
        RepAdapterConfig,
        FacTConfig,
        GPSConfig,
        VisualPromptTuningConfig,
    ],
    Field(discriminator="method"),
]
