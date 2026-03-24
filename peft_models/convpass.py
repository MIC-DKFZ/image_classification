import math
import torch
import torch.nn as nn
from functools import partial


MODEL_TO_ARCH_MAPPING = {
    "VisionTransformer": "VisionTransformer",
    "DinoVisionTransformer": "DinoVisionTransformer",
    "DINOv3ViTModel": "DINOv3ViTModel",
    "ViTModel": "ViTModel",
    "ViTMAEModel": "ViTModel",
}

# Reuses AdaptFormer's hook anchor points (parallel mode):
#   "parallel"   — pre-hook target: captures x before the second LayerNorm
#   "sequential" — post-hook target: fires after the full block for the residual add
MODULE_MAPPING = {
    "VisionTransformer":     {"block_modules": ["norm1", "attn", "norm2", "mlp"],                              "parallel": "norm2",          "sequential": "drop_path2"},
    "DinoVisionTransformer": {"block_modules": ["norm1", "attn", "norm2", "mlp"],                              "parallel": "norm2",          "sequential": "mlp.drop"},
    "DINOv3ViTModel":        {"block_modules": ["norm1", "attention", "norm2", "mlp"],                              "parallel": "norm2",          "sequential": "drop_path"},
    "ViTModel":              {"block_modules": ["layernorm_before", "attention", "layernorm_after", "intermediate"],  "parallel": "layernorm_after","sequential": "output.dropout"},
}


class ConvpassModule(nn.Module):
    """
    Convolutional bypass adapter.

    Forward path:
      1. Linear down-project all tokens (CLS + patches): D → bottleneck
      2. GELU activation
      3. Separate CLS and patch tokens
      4. Reshape patch tokens to 2D spatial grid [B, bottleneck, H, W]
      5. 3×3 Conv  (captures local spatial structure absent in ViT attention)
      6. Reshape back to sequence [B, N, bottleneck]
      7. Recombine with CLS
      8. Linear up-project: bottleneck → D

    Zero-initialised up-projection → ΔW = 0 at the start of training.
    CLS token bypasses the conv (down → act → up only), consistent with its
    non-spatial role.
    """

    def __init__(self, d_model: int, bottleneck: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1)
        self.up   = nn.Linear(bottleneck, d_model)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]  — L = CLS + (optional register tokens) + patch tokens
        # Patch tokens are always last; use the largest perfect square ≤ L-1 as N_patch.
        import math
        B, L, _ = x.shape
        H = math.isqrt(L - 1)        # e.g. isqrt(196)=14 (standard), isqrt(200)=14 (DINOv3)
        N_patch = H * H              # number of spatial patch tokens
        n_prefix = L - N_patch       # CLS + register tokens (1 standard, 5 for DINOv3)

        h = self.drop(self.act(self.down(x)))       # [B, L, bottleneck]

        prefix_h = h[:, :n_prefix, :]               # [B, n_prefix, bottleneck]
        patch_h  = h[:, n_prefix:, :]               # [B, N_patch,  bottleneck]

        # 2-D convolution on patch tokens
        patch_h = patch_h.permute(0, 2, 1).reshape(B, -1, H, H)     # [B, bottleneck, H, H]
        patch_h = self.conv(patch_h)                                   # [B, bottleneck, H, H]
        patch_h = patch_h.reshape(B, -1, N_patch).permute(0, 2, 1)   # [B, N_patch, bottleneck]

        h = torch.cat([prefix_h, patch_h], dim=1)  # [B, L, bottleneck]
        return self.up(h)                           # [B, L, D]


class Convpass:
    """
    Convpass (PETL-ViT, AAAI 2023): convolutional bypass adapter for ViT.

    Inserts a ConvpassModule in parallel with the FFN of every transformer
    block via forward hooks (no backbone modification). The spatial conv
    injects local inductive bias that self-attention lacks, making this
    adapter complementary to attention-based PEFT methods.

    Hook structure mirrors AdaptFormer-parallel:
      • pre-hook  on norm2 : captures x before the second LN, runs the conv bypass
      • post-hook on block : adds the bypass output to the block's final output

    Paper / repo: https://github.com/JieShibo/PETL-ViT
    """

    def __init__(
        self,
        bottleneck: int = 64,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        attach_convpass_with_hooks(self.model, bottleneck, dropout, freeze_backbone)


def attach_convpass_with_hooks(
    model: nn.Module,
    bottleneck: int,
    dropout: float,
    freeze_backbone: bool,
):
    target_arch = MODEL_TO_ARCH_MAPPING[model.__class__.__name__]
    mapping     = MODULE_MAPPING[target_arch]
    hooked      = []

    for blk in model.modules():
        if not all(hasattr(blk, a) for a in mapping["block_modules"]):
            continue

        # Infer hidden dim from the first block module (a LayerNorm)
        first_ln = _get_module(blk, mapping["block_modules"][0])
        ns = first_ln.normalized_shape
        d_model = ns[-1] if isinstance(ns, (list, tuple)) else int(ns)

        blk.convpass = ConvpassModule(d_model=d_model, bottleneck=bottleneck, dropout=dropout)
        blk._convpass_adapt_x = None

        if freeze_backbone:
            for p in blk.parameters():
                p.requires_grad = False
            for p in blk.convpass.parameters():
                p.requires_grad = True

        # PRE-HOOK on norm2: capture x before LayerNorm, run conv bypass (no residual)
        def _pre_hook(parent, mod, inputs):
            (x,) = inputs
            parent._convpass_adapt_x = parent.convpass(x)

        # POST-HOOK on block's last op: add conv bypass to final block output
        def _post_hook(parent, mod, inputs, output):
            ax = parent._convpass_adapt_x
            if ax is None or ax.shape[-1] != output.shape[-1]:
                return output
            parent._convpass_adapt_x = None
            return output + ax

        _get_module(blk, mapping["parallel"]).register_forward_pre_hook(
            partial(_pre_hook, blk)
        )
        _get_module(blk, mapping["sequential"]).register_forward_hook(
            partial(_post_hook, blk)
        )

        hooked.append(blk)

    if not hooked:
        raise RuntimeError("Convpass: no ViT-like blocks found in model.")


def _get_module(module: nn.Module, name: str) -> nn.Module:
    """Traverse a module hierarchy by dot-separated name and return the submodule."""
    current = module
    for attr in name.split("."):
        if attr in current._modules:
            current = current._modules[attr]
        else:
            current = getattr(current, attr)
    return current
