import torch
import torch.nn as nn
from types import SimpleNamespace
from functools import partial


MODEL_TO_ARCH_MAPPING = {
    "VisionTransformer": "VisionTransformer",
    "DinoVisionTransformer": "DinoVisionTransformer",
    "DINOv3ViTModel": "DINOv3ViTModel",
    "ViTModel": "ViTModel",
    "ViTMAEModel": "ViTModel",
}

MODULE_MAPPING = {
    "VisionTransformer": {"block_modules": ["norm1", "attn", "norm2", "mlp"], "sequential": "drop_path2", "parallel": "norm2"},
    "DinoVisionTransformer": {"block_modules": ["norm1", "attn", "norm2", "mlp"], "sequential": "mlp.drop", "parallel": "norm2"},
    "DINOv3ViTModel": {"block_modules": ["norm1", "attention", "norm2", "mlp"], "sequential": "drop_path", "parallel": "norm2"},
    "ViTModel": {"block_modules": ["layernorm_before", "attention", "layernorm_after", "intermediate"], "sequential": "output.dropout", "parallel": "layernorm_after"},
}


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

        return up


def _infer_block_dim(norm1: nn.Module) -> int:
    # timm blocks have norm1.normalized_shape
    nshape = getattr(norm1, "normalized_shape", None)  # Timm: norm1
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

    target_arch = MODEL_TO_ARCH_MAPPING[model.__class__.__name__]

    for blk in model.modules():
        # Heuristic for timm ViT blocks
        if not all(hasattr(blk, a) for a in MODULE_MAPPING[target_arch]["block_modules"]):
            continue

        dim = _infer_block_dim(get_module(blk, MODULE_MAPPING[target_arch]["block_modules"][0]))

        # Install adapter as a submodule on the parent block
        blk.adapter = AdapterModule(
            d_model=dim,
            bottleneck=bottleneck,
            dropout=dropout,
            init_option=init_option,
            adapter_scalar=adapter_scalar,
            adapter_layernorm_option=adapter_layernorm_option,
        )

        run_adapter = None
        if target_arch == "DINOv3ViTModel":
            run_adapter = False
            
        blk._adapter_ctx = SimpleNamespace(adapt_x=None, run_adapter=run_adapter, mode=mode)
        handles = []

        if mode == "sequential":
            # out <- adapter(out)  (post-hook on the block)
            def mlp_post_hook(parent, mod, inputs, output):
                if parent._adapter_ctx.run_adapter is None or parent._adapter_ctx.run_adapter == True:
                    output = parent.adapter(output)
                    parent._adapter_ctx.run_adapter = False
                else:
                    parent._adapter_ctx.run_adapter = True
                return output
            h = get_module(blk, MODULE_MAPPING[target_arch]["sequential"]).register_forward_hook(partial(mlp_post_hook, blk))
            handles.append(h)

        else:  # parallel
            # PRE-HOOK on norm2: capture x BEFORE norm2
            def norm2_pre_hook(parent, mod_norm2, inputs):
                (x_before_norm2,) = inputs
                parent._adapter_ctx.adapt_x = parent.adapter(x_before_norm2, add_residual=False)
                parent._adapter_norm2_done = True
            h1 = get_module(blk, MODULE_MAPPING[target_arch]["parallel"]).register_forward_pre_hook(partial(norm2_pre_hook, blk))
            handles.append(h1)

            # POST-HOOK on the block: add adapter contribution to final output
            def mlp_post_hook(parent, mod, inputs, output):
                ax = parent._adapter_ctx.adapt_x
                if ax is None or ax.shape[-1] != output.shape[-1]:  # ax.shape[-1] != output.shape[-1] -> Necessary in dinov3_reference as drop module is used twice in mlp module
                    return output
                parent._adapter_ctx.adapt_x = None
                output = output + ax
                return output
            h2 = get_module(blk, MODULE_MAPPING[target_arch]["sequential"]).register_forward_hook(partial(mlp_post_hook, blk))
            handles.append(h2)

        hooked.append((blk, handles))

    if not hooked:
        raise RuntimeError("No ViT-like blocks found.")

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if "adapter" in name:
                p.requires_grad = True

    return hooked


def get_module(module: nn.Module, name: str) -> nn.Module:
    """Traverse a module hierarchy by dot-separated name and return the submodule."""
    current = module
    for attr in name.split("."):
        # Try named_children / _modules first (safe for nn.Module containers)
        if attr in current._modules:
            current = current._modules[attr]
        else:
            # Fall back to normal attribute lookup (e.g. plain fields)
            current = getattr(current, attr)
    return current