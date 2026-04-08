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

# (dot-path within block, attr name to store DiffFitScaleLayer on the block)
DIFFFIT_TARGETS = {
    "VisionTransformer": [
        ("norm1",    "dff_norm1"),
        ("attn.qkv", "dff_qkv"),
        ("attn.proj","dff_proj"),
        ("norm2",    "dff_norm2"),
        ("mlp.fc1",  "dff_fc1"),
        ("mlp.fc2",  "dff_fc2"),
    ],
    "DinoVisionTransformer": [
        ("norm1",    "dff_norm1"),
        ("attn.qkv", "dff_qkv"),    # fused Q,K,V → dim = 3*D
        ("attn.proj","dff_proj"),
        ("norm2",    "dff_norm2"),
        ("mlp.fc1",  "dff_fc1"),
        ("mlp.fc2",  "dff_fc2"),
    ],
    "DINOv3ViTModel": [
        ("norm1",            "dff_norm1"),
        ("attention.q_proj", "dff_q"),
        ("attention.k_proj", "dff_k"),
        ("attention.v_proj", "dff_v"),
        ("attention.o_proj", "dff_proj"),
        ("norm2",            "dff_norm2"),
        ("mlp.up_proj",      "dff_fc1"),
        ("mlp.down_proj",    "dff_fc2"),
    ],
    "ViTModel": [
        ("layernorm_before",          "dff_norm1"),
        ("attention.attention.query", "dff_q"),
        ("attention.attention.key",   "dff_k"),
        ("attention.attention.value", "dff_v"),
        ("attention.output.dense",    "dff_proj"),
        ("layernorm_after",           "dff_norm2"),
        ("intermediate.dense",        "dff_fc1"),
        ("output.dense",              "dff_fc2"),
    ],
}


class DiffFit:
    """
    DiffFit (ICCV 2023): BitFit + per-channel learnable scale γ on sublayer outputs.

    Two complementary components:
      • Bias tuning  — all bias terms kept trainable (same as BitFit).
      • Scale tuning — a per-channel scale γ (init=1) is applied after each key
                       sublayer output via a forward hook: output = γ ⊙ output.

    Together these add slightly more capacity than BitFit alone while remaining
    far below LoRA / adapter methods in parameter count. The scale component
    can be merged into the preceding linear weight at inference (zero overhead).

    Paper: https://arxiv.org/abs/2304.06648
    """

    def __init__(self, *args, **kwargs):
        attach_difffit_with_hooks(self.model)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head", "dff_", "bias"]):
                param.requires_grad = True


class DiffFitScaleLayer(nn.Module):
    """Per-channel scale γ initialized to 1 (identity). No shift."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _infer_output_dim(module: nn.Module) -> int:
    if isinstance(module, nn.LayerNorm):
        ns = module.normalized_shape
        return ns[-1] if isinstance(ns, (list, tuple)) else int(ns)
    if isinstance(module, nn.Linear):
        return module.weight.shape[0]
    raise TypeError(f"DiffFit: cannot infer output dim for {type(module).__name__}")


def attach_difffit_with_hooks(model: nn.Module):
    target_arch = MODEL_TO_ARCH_MAPPING[model.__class__.__name__]
    block_markers = BLOCK_MARKERS[target_arch]
    difffit_targets = DIFFFIT_TARGETS[target_arch]

    hooked = []

    for blk in model.modules():
        if not all(hasattr(blk, m) for m in block_markers):
            continue

        for dot_path, attr_name in difffit_targets:
            try:
                submod = _get_module(blk, dot_path)
            except (AttributeError, KeyError):
                continue

            scale_layer = DiffFitScaleLayer(_infer_output_dim(submod))
            setattr(blk, attr_name, scale_layer)

            def _make_hook(scale):
                def hook(mod, inputs, output):
                    if isinstance(output, tuple):
                        return (scale(output[0]),) + output[1:]
                    return scale(output)
                return hook

            submod.register_forward_hook(_make_hook(scale_layer))

        hooked.append(blk)

    if not hooked:
        raise RuntimeError("DiffFit: no transformer blocks found in model.")


def _get_module(module: nn.Module, name: str) -> nn.Module:
    """Traverse a module hierarchy by dot-separated name and return the submodule."""
    current = module
    for attr in name.split("."):
        if attr in current._modules:
            current = current._modules[attr]
        else:
            current = getattr(current, attr)
    return current
