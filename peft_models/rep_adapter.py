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

# (dot-path within block, attr name to store RepAdapterModule on the block)
REPA_TARGETS = {
    "VisionTransformer": [
        ("attn.qkv",  "repa_qkv"),   # fused Q,K,V
        ("attn.proj", "repa_proj"),
        ("mlp.fc1",   "repa_fc1"),
        ("mlp.fc2",   "repa_fc2"),
    ],
    "DinoVisionTransformer": [
        ("attn.q_proj", "repa_q"),
        ("attn.k_proj", "repa_k"),
        ("attn.v_proj", "repa_v"),
        ("attn.proj",   "repa_proj"),
        ("mlp.fc1",     "repa_fc1"),
        ("mlp.fc2",     "repa_fc2"),
    ],
    "DINOv3ViTModel": [
        ("attention.q_proj", "repa_q"),
        ("attention.k_proj", "repa_k"),
        ("attention.v_proj", "repa_v"),
        ("attention.o_proj", "repa_proj"),
        ("mlp.up_proj",      "repa_fc1"),
        ("mlp.down_proj",    "repa_fc2"),
    ],
    "ViTModel": [
        ("attention.attention.query", "repa_q"),
        ("attention.attention.key",   "repa_k"),
        ("attention.attention.value", "repa_v"),
        ("attention.output.dense",    "repa_proj"),
        ("intermediate.dense",        "repa_fc1"),
        ("output.dense",              "repa_fc2"),
    ],
}


class RepAdapterModule(nn.Module):
    """
    Purely linear bottleneck: down → GroupNorm → up → learnable scale.

    No activation is used intentionally: keeping the adapter linear makes
    it structurally reparameterizable — at inference the GroupNorm statistics
    can be absorbed into the linear weights and the whole adapter folded into
    the original weight matrix (W_merged = W₀ + scale · W_up W_down_reparam).

    GroupNorm normalises the bottleneck features across channel groups,
    stabilising training without introducing batch-size dependency (unlike BN).
    """

    def __init__(self, d_in: int, d_out: int, bottleneck: int, groups: int = 2, scale_init: float = 1e-3):
        super().__init__()

        # Ensure groups evenly divides bottleneck
        while bottleneck % groups != 0 and groups > 1:
            groups -= 1

        self.down  = nn.Linear(d_in, bottleneck)
        self.gn    = nn.GroupNorm(num_groups=groups, num_channels=bottleneck)
        self.up    = nn.Linear(bottleneck, d_out)
        self.scale = nn.Parameter(torch.full((1,), scale_init))

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)    # zero-init → ΔW = 0 at start
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d_in]
        h = self.down(x)                         # [B, N, bottleneck]
        h = self.gn(h.transpose(-2, -1))         # GroupNorm expects [B, C, *]
        h = h.transpose(-2, -1)                  # [B, N, bottleneck]
        return self.up(h) * self.scale           # [B, N, d_out]


class RepAdapter:
    """
    RepAdapter (arXiv 2302.08106, 2023): structurally reparameterizable adapter.

    A purely linear bottleneck (down → GroupNorm → up) is added in parallel
    to each target linear layer via a forward post-hook:

        output = W₀ · x  +  scale · up(GN(down(x)))

    Because no activation is used, the full adapter path is linear after
    GroupNorm statistics are absorbed into the weights. This allows merging
    the adapter into W₀ at inference with zero runtime overhead:

        W_merged = W₀ + scale · W_up · W_down_reparam

    where W_down_reparam absorbs the GroupNorm affine transform.

    Paper: https://arxiv.org/abs/2302.08106
    """

    def __init__(
        self,
        repadapter_bottleneck: int = 8,
        repadapter_groups: int = 2,
        repadapter_scale_init: float = 1e-3,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        attach_repadapters_with_hooks(
            self.model,
            bottleneck=repadapter_bottleneck,
            groups=repadapter_groups,
            scale_init=repadapter_scale_init,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head", "repa_"]):
                param.requires_grad = True


def attach_repadapters_with_hooks(
    model: nn.Module,
    bottleneck: int,
    groups: int,
    scale_init: float,
):
    target_arch = MODEL_TO_ARCH_MAPPING[model.__class__.__name__]
    block_markers = BLOCK_MARKERS[target_arch]
    repa_targets = REPA_TARGETS[target_arch]

    hooked = []

    for blk in model.modules():
        if not all(hasattr(blk, m) for m in block_markers):
            continue

        for dot_path, attr_name in repa_targets:
            try:
                submod = _get_module(blk, dot_path)
            except (AttributeError, KeyError):
                continue

            if not isinstance(submod, nn.Linear):
                continue

            d_in  = submod.weight.shape[1]   # input features of the linear layer
            d_out = submod.weight.shape[0]   # output features of the linear layer
            adapter = RepAdapterModule(d_in, d_out, bottleneck, groups, scale_init)
            setattr(blk, attr_name, adapter)   # registers in blk._modules

            def _make_hook(adp):
                def hook(mod, inputs, output):
                    if isinstance(output, tuple):
                        return (output[0] + adp(inputs[0]),) + output[1:]
                    return output + adp(inputs[0])
                return hook

            submod.register_forward_hook(_make_hook(adapter))

        hooked.append(blk)

    if not hooked:
        raise RuntimeError("RepAdapter: no transformer blocks found in model.")


def _get_module(module: nn.Module, name: str) -> nn.Module:
    """Traverse a module hierarchy by dot-separated name and return the submodule."""
    current = module
    for attr in name.split("."):
        if attr in current._modules:
            current = current._modules[attr]
        else:
            current = getattr(current, attr)
    return current
