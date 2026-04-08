"""Optimizer factory and layer-wise LR decay for ViT backbones."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any

import torch
import torch.nn as nn

from glovita.configs.optimizer import OptimizerConfig


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Construct an optimizer from *config*, respecting parameter grouping rules.

    Three mutually exclusive grouping strategies are supported:
    1. **layer_wise_lr_decay** — BERT-style decaying LR per ViT block depth.
    2. **undecay_norm** — zero weight-decay for bias / norm / bn parameters.
    3. **default** — all trainable parameters in a single group.
    """
    if config.layer_wise_lr_decay is not None:
        # layer_wise_lr_decay requires access to model.model (backbone ViT)
        backbone = getattr(model, "model", model)
        params = _get_layerwise_lr_params(
            backbone,
            base_lr=config.lr,
            weight_decay=config.weight_decay,
            layer_decay=config.layer_wise_lr_decay,
        )
    elif config.undecay_norm:
        model_params, norm_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("norm", "bias", "bn")):
                norm_params.append(p)
            else:
                model_params.append(p)
        params: list[dict[str, Any]] = [
            {"params": model_params, "weight_decay": config.weight_decay},
            {"params": norm_params, "weight_decay": 0.0},
        ]
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    kwargs: dict[str, Any] = {"lr": config.lr, "weight_decay": config.weight_decay}

    if config.name == "SGD":
        kwargs["momentum"] = 0.9
        kwargs["nesterov"] = config.nesterov
        return torch.optim.SGD(params, **kwargs)
    if config.name == "Adam":
        return torch.optim.Adam(params, **kwargs)
    if config.name == "AdamW":
        return torch.optim.AdamW(params, **kwargs)
    if config.name == "RMSprop":
        from timm.optim import RMSpropTF
        return RMSpropTF(params, **kwargs)
    if config.name == "Madgrad":
        from madgrad import MADGRAD
        kwargs["momentum"] = 0.9
        return MADGRAD(params, **kwargs)
    if config.name == "Muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "Optimizer 'Muon' was requested, but torch.optim.Muon is not "
                f"available in the installed torch build ({torch.__version__})."
            )
        kwargs["momentum"] = config.muon_momentum
        kwargs["nesterov"] = config.nesterov
        kwargs["ns_coefficients"] = config.muon_ns_coefficients
        kwargs["eps"] = config.muon_eps
        kwargs["ns_steps"] = config.muon_ns_steps
        kwargs["adjust_lr_fn"] = config.muon_adjust_lr_fn
        return torch.optim.Muon(params, **kwargs)

    raise ValueError(f"Unknown optimizer: {config.name!r}")


def _get_layerwise_lr_params(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> list[dict[str, Any]]:
    """BERT-style layer-wise LR decay for a timm ViT.

    Depth assignment:
      patch_embed          → depth 1
      blocks.i             → depth i + 2
      fc_norm / norm       → depth n_blocks + 2
      head / cls_head      → depth n_blocks + 3

    LR = base_lr × layer_decay^(max_depth - depth)
    """
    if not hasattr(model, "blocks"):
        raise AttributeError(
            "layer_wise_lr_decay requires a backbone with a '.blocks' attribute (timm ViT)."
        )
    n_blocks = len(model.blocks)

    key2depth: OrderedDict[str, int] = OrderedDict()
    key2depth["patch_embed"] = 1
    for i in range(n_blocks):
        key2depth[f"blocks.{i}."] = i + 2
    key2depth["fc_norm"] = n_blocks + 2
    key2depth["norm"] = n_blocks + 2
    key2depth["head"] = n_blocks + 3

    max_depth = n_blocks + 3
    lr_groups: defaultdict[float, list] = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for key, depth in key2depth.items():
            if key in name:
                lr = base_lr * (layer_decay ** (max_depth - depth))
                lr_groups[lr].append(param)
                break
        else:
            lr_groups[base_lr].append(param)

    return [
        {"params": ps, "lr": lr, "weight_decay": weight_decay}
        for lr, ps in sorted(lr_groups.items())
    ]
