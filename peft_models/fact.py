import math
import torch
import torch.nn as nn


MODEL_TO_ARCH_MAPPING = {
    "VisionTransformer": "VisionTransformer",
    "DinoVisionTransformer": "DinoVisionTransformer",
    "DINOv3ViTModel": "DINOv3ViTModel",
    "ViTModel": "ViTModel",
    "ViTMAEModel": "ViTModel",
}

BLOCK_MARKERS = {
    "VisionTransformer":     ["norm1", "attn", "norm2", "mlp"],
    "DinoVisionTransformer": ["norm1", "attn", "norm2", "mlp"],
    "DINOv3ViTModel":        ["norm1", "attention", "norm2", "mlp"],
    "ViTModel":              ["layernorm_before", "attention", "layernorm_after", "intermediate"],
}

# (logical_type, dot_path_within_block)
# Layers sharing the same logical_type share one (up, down) pair of factor matrices.
FACT_TARGETS = {
    "VisionTransformer": [
        ("qkv",  "attn.qkv"),   # fused Q,K,V; shared up/down dim = (3D, D)
        ("proj", "attn.proj"),
        ("fc1",  "mlp.fc1"),
        ("fc2",  "mlp.fc2"),
    ],
    "DinoVisionTransformer": [
        ("q",    "attn.q_proj"),
        ("k",    "attn.k_proj"),
        ("v",    "attn.v_proj"),
        ("proj", "attn.proj"),
        ("fc1",  "mlp.fc1"),
        ("fc2",  "mlp.fc2"),
    ],
    "DINOv3ViTModel": [
        ("q",    "attention.q_proj"),
        ("k",    "attention.k_proj"),
        ("v",    "attention.v_proj"),
        ("proj", "attention.o_proj"),
        ("fc1",  "mlp.up_proj"),
        ("fc2",  "mlp.down_proj"),
    ],
    "ViTModel": [
        ("q",    "attention.attention.query"),
        ("k",    "attention.attention.key"),
        ("v",    "attention.attention.value"),
        ("proj", "attention.output.dense"),
        ("fc1",  "intermediate.dense"),
        ("fc2",  "output.dense"),
    ],
}


class FacTContainer(nn.Module):
    """
    Holds all trainable FacT parameters outside the frozen backbone:
      - One shared (up, down) linear pair per weight type
      - One per-layer scale vector sᵢ ∈ R^r per weight type
    """

    def __init__(self, dims: dict, n_layers: int, r: int):
        super().__init__()

        for lt, (d_out, d_in) in dims.items():
            up = nn.Linear(d_in, r, bias=False)
            down = nn.Linear(r, d_out, bias=False)
            nn.init.kaiming_uniform_(up.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            setattr(self, f"up_{lt}", up)
            setattr(self, f"down_{lt}", down)
            # Zero-init scales → ΔWᵢ = 0 at initialisation for all layers
            setattr(self, f"scales_{lt}", nn.Parameter(torch.zeros(n_layers, r)))

    def compute(self, logical_type: str, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        up    = getattr(self, f"up_{logical_type}")
        down  = getattr(self, f"down_{logical_type}")
        scale = getattr(self, f"scales_{logical_type}")[layer_idx]  # [r]
        return down(up(x) * scale)                                   # [B, N, d_out]


class FacT:
    """
    FacT — Factor-Tuning (AAAI 2023).

    Decomposes ALL weight updates across ALL layers of the same type into a
    shared compact factorisation:

        ΔWᵢ · x  =  down · (sᵢ ⊙ (up · x))

    where:
      • up   ∈ R^{r × d_in}  — shared across every layer of that weight type
      • down ∈ R^{d_out × r} — shared across every layer of that weight type
      • sᵢ  ∈ R^r            — layer-specific scale vector (the only per-layer params)

    The shared up/down matrices amortise parameter cost across depth; the entire
    ViT-B/16 update for r=4 costs ~49K parameters vs ~1.2M for LoRA at rank 16.

    All FacT parameters live in self.fact_container (outside self.model), so
    freezing self.model.parameters() is sufficient to freeze the backbone.

    Paper: https://arxiv.org/abs/2212.03145
    """

    def __init__(self, fact_r, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fact_container = _attach_fact(self.model, fact_r)

        for param in self.model.parameters():
            param.requires_grad = False
        # self.fact_container and head live outside self.model → remain trainable


def _attach_fact(model: nn.Module, r: int) -> FacTContainer:
    target_arch = MODEL_TO_ARCH_MAPPING[model.__class__.__name__]
    block_markers = BLOCK_MARKERS[target_arch]
    fact_targets = FACT_TARGETS[target_arch]

    # Pass 1: collect blocks in order; infer dims from the first block
    blocks = []
    dims = {}

    for blk in model.modules():
        if not all(hasattr(blk, m) for m in block_markers):
            continue
        if not blocks:
            for lt, dot_path in fact_targets:
                try:
                    submod = _get_module(blk, dot_path)
                    if isinstance(submod, nn.Linear):
                        dims[lt] = (submod.weight.shape[0], submod.weight.shape[1])
                except (AttributeError, KeyError):
                    pass
        blocks.append(blk)

    if not blocks:
        raise RuntimeError("FacT: no transformer blocks found in model.")

    container = FacTContainer(dims=dims, n_layers=len(blocks), r=r)

    # Pass 2: attach one forward hook per (block, weight_type) pair
    for layer_idx, blk in enumerate(blocks):
        for lt, dot_path in fact_targets:
            if lt not in dims:
                continue
            try:
                submod = _get_module(blk, dot_path)
            except (AttributeError, KeyError):
                continue

            def _make_hook(ct, logical_type, li):
                def hook(_, inputs, output):
                    delta = ct.compute(logical_type, li, inputs[0])
                    if isinstance(output, tuple):
                        return (output[0] + delta,) + output[1:]
                    return output + delta
                return hook

            submod.register_forward_hook(_make_hook(container, lt, layer_idx))

    return container


def _get_module(module: nn.Module, name: str) -> nn.Module:
    """Traverse a module hierarchy by dot-separated name and return the submodule."""
    current = module
    for attr in name.split("."):
        if attr in current._modules:
            current = current._modules[attr]
        else:
            current = getattr(current, attr)
    return current
