import torch
import torch.nn as nn
from types import SimpleNamespace
from functools import partial
from typing import List


class AdaptFormer:
    def __init__(self, 
                 mode: str = "sequential",     # "sequential" | "parallel"
                 bottleneck: int = 64,
                 dropout: float = 0.0,
                 init_option: str = "lora",
                 adapter_scalar: str = "1.0",  # or "learnable_scalar"
                 adapter_layernorm_option: str = "in",
                 freeze_backbone: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        attach_adapters_with_hooks(self.model, mode, bottleneck, dropout, init_option, adapter_scalar, adapter_layernorm_option, freeze_backbone)


class AdapterModule(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        if d_model is None and (config is None or not hasattr(config, "d_model")):
            raise ValueError("Adapter: need d_model or config.d_model.")
        if bottleneck is None and (config is None or not hasattr(config, "attn_bn")):
            raise ValueError("Adapter: need bottleneck or config.attn_bn.")

        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        elif init_option == "bert":
            raise NotImplementedError

    def forward(self, x, add_residual=True, residual=None):
        x_is_list = False
        if isinstance(x, List) and len(x) == 1: # DINOv3 fix
            x = x[0]
            x_is_list = True
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        up = up + residual if add_residual else up

        if x_is_list:
            up = [up]

        return up


def _infer_block_dim(block: nn.Module) -> int:
    # timm blocks have norm1.normalized_shape
    nshape = getattr(block.norm1, "normalized_shape", None)
    if nshape is None:
        raise RuntimeError("Cannot infer block dim; provide d_model explicitly.")
    return nshape if isinstance(nshape, int) else nshape[-1]


def attach_adapters_with_hooks(
    model: nn.Module,
    mode: str = "parallel",      # "sequential" | "parallel"
    bottleneck: int = 64,
    dropout: float = 0.0,
    init_option: str = "lora",
    adapter_scalar: str = "1.0",
    adapter_layernorm_option: str = "in",
    freeze_backbone: bool = True,
):
    assert mode in ("sequential", "parallel")
    hooked = []

    for blk in model.modules():
        # Heuristic for timm ViT blocks
        if not all(hasattr(blk, a) for a in ("norm1", "attn", "mlp", "norm2")):
            continue

        dim = _infer_block_dim(blk)

        # Install adapter as a submodule on the parent block
        blk.adapter = AdapterModule(
            d_model=dim,
            bottleneck=bottleneck,
            dropout=dropout,
            init_option=init_option,
            adapter_scalar=adapter_scalar,
            adapter_layernorm_option=adapter_layernorm_option,
        )

        if freeze_backbone:
            for p in blk.parameters():
                p.requires_grad = False
            for p in blk.adapter.parameters():
                p.requires_grad = True

        blk._adapter_ctx = SimpleNamespace(adapt_x=None, mode=mode)
        handles = []

        if mode == "sequential":
            # out <- adapter(out)  (post-hook on the block)
            def block_post_hook(parent, mod, inputs, output):
                return parent.adapter(output)
            h = blk.register_forward_hook(partial(block_post_hook, blk))
            handles.append(h)

        else:  # parallel
            # PRE-HOOK on norm2: capture x BEFORE norm2
            def norm2_pre_hook(parent, mod_norm2, inputs):
                (x_before_norm2,) = inputs
                parent._adapter_ctx.adapt_x = parent.adapter(x_before_norm2, add_residual=False)
            h1 = blk.norm2.register_forward_pre_hook(partial(norm2_pre_hook, blk))
            handles.append(h1)

            # POST-HOOK on the block: add adapter contribution to final output
            def block_post_hook(parent, mod, inputs, output):
                ax = parent._adapter_ctx.adapt_x
                parent._adapter_ctx.adapt_x = None
                return output + ax if ax is not None else output
            h2 = blk.register_forward_hook(partial(block_post_hook, blk))
            handles.append(h2)

        hooked.append((blk, handles))

    if not hooked:
        raise RuntimeError("No ViT-like blocks found (need norm1/attn/mlp/norm2).")
    return hooked
