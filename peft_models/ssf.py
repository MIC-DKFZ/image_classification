import torch
import torch.nn as nn


MODEL_TO_ARCH_MAPPING = {
    "VisionTransformer": "VisionTransformer",
    "DinoVisionTransformer": "DinoVisionTransformer",
    "DINOv3ViTModel": "DINOv3ViTModel",
    "ViTModel": "ViTModel",
    "ViTMAEModel": "ViTModel",
}

# Top-level attributes that uniquely identify a transformer block
BLOCK_MARKERS = {
    "VisionTransformer":      ["norm1", "attn", "norm2", "mlp"],
    "DinoVisionTransformer":  ["norm1", "attn", "norm2", "mlp"],
    "DINOv3ViTModel":         ["norm1", "attention", "norm2", "mlp"],
    "ViTModel":               ["layernorm_before", "attention", "layernorm_after", "intermediate"],
}

# (dot-path within block, attr name to store SSFLayer on the block)
SSF_TARGETS = {
    "VisionTransformer": [
        ("norm1",    "ssf_norm1"),
        ("attn.qkv", "ssf_qkv"),    # fused Q,K,V → dim = 3*D
        ("attn.proj","ssf_proj"),
        ("norm2",    "ssf_norm2"),
        ("mlp.fc1",  "ssf_fc1"),    # pre-activation; dim = mlp_ratio * D
        ("mlp.fc2",  "ssf_fc2"),
    ],
    "DinoVisionTransformer": [
        ("norm1",       "ssf_norm1"),
        ("attn.q_proj", "ssf_q"),
        ("attn.k_proj", "ssf_k"),
        ("attn.v_proj", "ssf_v"),
        ("attn.proj",   "ssf_proj"),
        ("norm2",       "ssf_norm2"),
        ("mlp.fc1",     "ssf_fc1"),
        ("mlp.fc2",     "ssf_fc2"),
    ],
    "DINOv3ViTModel": [
        ("norm1",            "ssf_norm1"),
        ("attention.q_proj", "ssf_q"),
        ("attention.k_proj", "ssf_k"),
        ("attention.v_proj", "ssf_v"),
        ("attention.o_proj", "ssf_proj"),
        ("norm2",            "ssf_norm2"),
        ("mlp.up_proj",      "ssf_fc1"),
        ("mlp.down_proj",    "ssf_fc2"),
    ],
    "ViTModel": [
        ("layernorm_before",          "ssf_norm1"),
        ("attention.attention.query", "ssf_q"),
        ("attention.attention.key",   "ssf_k"),
        ("attention.attention.value", "ssf_v"),
        ("attention.output.dense",    "ssf_proj"),
        ("layernorm_after",           "ssf_norm2"),
        ("intermediate.dense",        "ssf_fc1"),
        ("output.dense",              "ssf_fc2"),
    ],
}


class SSF:
    """
    Scale & Shift Features (SSF, NeurIPS 2022).

    Learns a per-channel scale γ and shift β applied after each key operation
    in every transformer block:  SSF(x) = γ ⊙ x + β

    All backbone parameters are frozen; only SSF parameters and the head
    are trained. SSF parameters can be merged into preceding linear/norm
    weights at inference for zero overhead.

    Paper: https://arxiv.org/abs/2210.08823
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        attach_ssf_with_hooks(self.model)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head", "ssf_"]):
                param.requires_grad = True


class SSFLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


def _infer_output_dim(module: nn.Module) -> int:
    if isinstance(module, nn.LayerNorm):
        ns = module.normalized_shape
        return ns[-1] if isinstance(ns, (list, tuple)) else int(ns)
    if isinstance(module, nn.Linear):
        return module.weight.shape[0]
    raise TypeError(f"SSF: cannot infer output dim for {type(module).__name__}")


def attach_ssf_with_hooks(model: nn.Module):
    target_arch = MODEL_TO_ARCH_MAPPING[model.__class__.__name__]
    block_markers = BLOCK_MARKERS[target_arch]
    ssf_targets = SSF_TARGETS[target_arch]

    hooked = []

    for blk in model.modules():
        if not all(hasattr(blk, m) for m in block_markers):
            continue

        for dot_path, attr_name in ssf_targets:
            try:
                submod = _get_module(blk, dot_path)
            except (AttributeError, KeyError):
                continue

            ssf_layer = SSFLayer(_infer_output_dim(submod))
            # Registers ssf_layer in blk._modules so it appears in named_parameters()
            setattr(blk, attr_name, ssf_layer)

            def _make_hook(ssf):
                def hook(mod, inputs, output):
                    if isinstance(output, tuple):
                        return (ssf(output[0]),) + output[1:]
                    return ssf(output)
                return hook

            submod.register_forward_hook(_make_hook(ssf_layer))

        hooked.append(blk)

    if not hooked:
        raise RuntimeError("SSF: no transformer blocks found in model.")


def _get_module(module: nn.Module, name: str) -> nn.Module:
    """Traverse a module hierarchy by dot-separated name and return the submodule."""
    current = module
    for attr in name.split("."):
        if attr in current._modules:
            current = current._modules[attr]
        else:
            current = getattr(current, attr)
    return current
