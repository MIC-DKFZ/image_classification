"""PEFT registry: single entry point that applies a PEFT method to a model.

The function ``apply_peft(model, config)`` dispatches on ``config.method`` and
calls the corresponding setup function.  Each setup function may:
  - Freeze / unfreeze parameters on the existing model (most methods).
  - Wrap ``model.model`` with a HuggingFace PEFT wrapper (LoRA family).
  - Insert new adapter modules into ``model.model`` (AdaptFormer, ConvPass, VPT).

All functions receive the *full model* object (e.g. ``TimmModel`` or
``TransformerModel``) so they can access ``model.model`` (the backbone) as well
as ``model.cls_head``.
"""
from __future__ import annotations

import torch.nn as nn

from glovita.configs.peft import (
    AdaLoraConfig,
    AdaptFormerConfig,
    BitFitConfig,
    BOFTConfig,
    ConvpassConfig,
    DiffFitConfig,
    FacTConfig,
    FourierFTConfig,
    FullFinetuningConfig,
    GPSConfig,
    IA3Config,
    LinearProbingConfig,
    LNTuningConfig,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    PartialFinetuningConfig,
    PeftConfig,
    RepAdapterConfig,
    SSFConfig,
    VeRAConfig,
    VisualPromptTuningConfig,
)


def apply_peft(model: nn.Module, config: PeftConfig) -> nn.Module:
    """Apply the PEFT method described by *config* to *model* in-place.

    Returns the (possibly wrapped) model.  The caller should use the returned
    object for training, not the original reference, because HuggingFace PEFT
    wrappers replace ``model.model``.
    """
    method = config.method

    if isinstance(config, LinearProbingConfig):
        _freeze_all_except_head(model)

    elif isinstance(config, FullFinetuningConfig):
        pass  # all parameters remain trainable

    elif isinstance(config, PartialFinetuningConfig):
        _apply_partial_finetuning(model, config)

    elif isinstance(config, BitFitConfig):
        _freeze_all_except(model, keep=["bias", "cls_head"])

    elif isinstance(config, LNTuningConfig):
        _freeze_all_except(model, keep=["norm", "ln", "layernorm", "cls_head"])

    elif isinstance(config, SSFConfig):
        from glovita.models.peft.ssf import SSF as SSFMixin
        SSFMixin.__init__(model)

    elif isinstance(config, DiffFitConfig):
        from glovita.models.peft.difffit import DiffFit as DiffFitMixin
        DiffFitMixin.__init__(model)

    elif isinstance(config, LoraConfig):
        from glovita.models.peft.lora import LoRA as LoRAMixin
        LoRAMixin.__init__(
            model,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
            use_dora=config.use_dora,
            use_rslora=config.use_rslora,
            init_lora_weights=config.init_lora_weights,
            lora_bias=config.lora_bias,
        )

    elif isinstance(config, AdaLoraConfig):
        from glovita.models.peft.adalora import AdaLoRA as AdaLoRAMixin
        AdaLoRAMixin.__init__(
            model,
            adalora_rank=config.adalora_rank,
            adalora_init_rank=config.adalora_init_rank,
            adalora_alpha=config.adalora_alpha,
            adalora_dropout=config.adalora_dropout,
            adalora_target_modules=config.adalora_target_modules,
            use_rslora=config.use_rslora,
            init_lora_weights=config.init_lora_weights,
            lora_bias=config.lora_bias,
            adalora_orth_reg_weight=config.adalora_orth_reg_weight,
            adalora_beta1=config.adalora_beta1,
            adalora_beta2=config.adalora_beta2,
            adalora_tinit=config.adalora_tinit,
            adalora_deltaT=config.adalora_deltaT,
        )

    elif isinstance(config, LoHaConfig):
        from glovita.models.peft.loha import LoHa as LoHaMixin
        LoHaMixin.__init__(
            model,
            loha_rank=config.loha_rank,
            loha_alpha=config.loha_alpha,
            loha_dropout=config.loha_dropout,
            loha_rank_dropout=config.loha_rank_dropout,
            loha_target_modules=config.loha_target_modules,
        )

    elif isinstance(config, LoKrConfig):
        from glovita.models.peft.lokr import LoKr as LoKrMixin
        LoKrMixin.__init__(
            model,
            lokr_rank=config.lokr_rank,
            lokr_alpha=config.lokr_alpha,
            lokr_dropout=config.lokr_dropout,
            lokr_rank_dropout=config.lokr_rank_dropout,
            lokr_decompose_factor=config.lokr_decompose_factor,
            lokr_target_modules=config.lokr_target_modules,
        )

    elif isinstance(config, OFTConfig):
        from glovita.models.peft.oft import OFT as OFTMixin
        OFTMixin.__init__(
            model,
            oft_r=config.oft_r,
            oft_dropout=config.oft_dropout,
            oft_target_modules=config.oft_target_modules,
            oft_coft=config.oft_coft,
            oft_eps=config.oft_eps,
            oft_bias=config.oft_bias,
            oft_use_cayley_neumann=config.oft_use_cayley_neumann,
        )

    elif isinstance(config, BOFTConfig):
        from glovita.models.peft.boft import BOFT as BOFTMixin
        BOFTMixin.__init__(
            model,
            boft_block_size=config.boft_block_size,
            boft_n_butterfly_factor=config.boft_n_butterfly_factor,
            boft_dropout=config.boft_dropout,
            boft_target_modules=config.boft_target_modules,
            boft_bias=config.boft_bias,
        )

    elif isinstance(config, VeRAConfig):
        from glovita.models.peft.vera import Vera as VeraMixin
        VeraMixin.__init__(
            model,
            vera_rank=config.vera_rank,
            vera_dropout=config.vera_dropout,
            vera_target_modules=config.vera_target_modules,
            vera_projection_prng_key=config.vera_projection_prng_key,
        )

    elif isinstance(config, IA3Config):
        from glovita.models.peft.ia3 import IA3 as IA3Mixin
        IA3Mixin.__init__(model)

    elif isinstance(config, FourierFTConfig):
        from glovita.models.peft.fourierft import FourierFT as FourierFTMixin
        FourierFTMixin.__init__(
            model,
            fourierft_n_frequency=config.fourierft_n_frequency,
            fourierft_scaling=config.fourierft_scaling,
            fourierft_target_modules=config.fourierft_target_modules,
        )

    elif isinstance(config, AdaptFormerConfig):
        from glovita.models.peft.adapt_former import AdaptFormer as AdaptFormerMixin
        AdaptFormerMixin.__init__(
            model,
            mode=config.mode,
            bottleneck=config.bottleneck,
            dropout=config.dropout,
            init_option=config.init_option,
            adapter_scalar=config.adapter_scalar,
            adapter_layernorm_option=config.adapter_layernorm_option,
            freeze_backbone=config.freeze_backbone,
        )

    elif isinstance(config, ConvpassConfig):
        from glovita.models.peft.convpass import Convpass as ConvpassMixin
        ConvpassMixin.__init__(
            model,
            bottleneck=config.bottleneck,
            dropout=config.dropout,
            freeze_backbone=config.freeze_backbone,
        )

    elif isinstance(config, RepAdapterConfig):
        from glovita.models.peft.rep_adapter import RepAdapter as RepAdapterMixin
        RepAdapterMixin.__init__(
            model,
            repadapter_bottleneck=config.repadapter_bottleneck,
            repadapter_groups=config.repadapter_groups,
            repadapter_scale_init=config.repadapter_scale_init,
        )

    elif isinstance(config, FacTConfig):
        from glovita.models.peft.fact import FacT as FacTMixin
        FacTMixin.__init__(model, fact_r=config.fact_r)

    elif isinstance(config, GPSConfig):
        from glovita.models.peft.gps import GPS as GPSMixin
        GPSMixin.__init__(model, gps_percent=config.gps_percent)

    elif isinstance(config, VisualPromptTuningConfig):
        from glovita.models.peft.visual_prompt_tuning import VisualPromptTuning as VPTMixin
        VPTMixin.__init__(
            model,
            num_tokens=config.num_tokens,
            deep=config.deep,
            project_dim=config.project_dim,
            dropout=config.dropout,
            init_scale=config.init_scale,
            deep_layers=config.deep_layers,
        )

    else:
        raise ValueError(f"Unknown PEFT method: {method!r}")

    return model


# ---------------------------------------------------------------------------
# Shared freeze / unfreeze helpers
# ---------------------------------------------------------------------------

_HEAD_SUBSTRINGS = ("head", "classifier", "cls_head")


def _freeze_all_except_head(model: nn.Module) -> None:
    """Freeze everything except classification head parameters."""
    for name, param in model.named_parameters():
        param.requires_grad = any(sub in name for sub in _HEAD_SUBSTRINGS)


def _freeze_all_except(model: nn.Module, keep: list[str]) -> None:
    """Freeze all parameters whose name does not contain any string in *keep*."""
    keep_subs = tuple(keep) + _HEAD_SUBSTRINGS
    for name, param in model.named_parameters():
        param.requires_grad = any(sub in name for sub in keep_subs)


def _apply_partial_finetuning(model: nn.Module, config: PartialFinetuningConfig) -> None:
    """Freeze the first ``num_frozen_layers`` transformer blocks; unfreeze the rest."""
    backbone = getattr(model, "model", model)

    if not hasattr(backbone, "blocks"):
        raise AttributeError(
            "PartialFinetuning requires a backbone with a '.blocks' attribute (timm ViT)."
        )

    # Start fully frozen
    for param in backbone.parameters():
        param.requires_grad = False

    # Unfreeze blocks from num_frozen_layers onwards
    for i, block in enumerate(backbone.blocks):
        if i >= config.num_frozen_layers:
            for param in block.parameters():
                param.requires_grad = True

    # Always unfreeze norm + head
    for name, param in model.named_parameters():
        if any(sub in name for sub in ("norm", "fc_norm") + _HEAD_SUBSTRINGS):
            param.requires_grad = True
