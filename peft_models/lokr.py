from peft import get_peft_model, LoKrConfig
from peft.tuners.tuners_utils import BaseTunerLayer


def _patch_peft_linear_compat(peft_model):
    """Proxy in_features/out_features onto PEFT-wrapped layers that omit them.

    LoHa and LoKr wrappers (unlike LoRA/BOFT) do not delegate nn.Linear attrs
    via __getattr__. Some backbones (e.g. DINOv3) call self.qkv.in_features
    directly inside their forward pass, so patching restores compatibility.
    """
    for module in peft_model.modules():
        if isinstance(module, BaseTunerLayer) and hasattr(module, "base_layer"):
            base = module.base_layer
            if not hasattr(module, "in_features") and hasattr(base, "in_features"):
                module.in_features = base.in_features
            if not hasattr(module, "out_features") and hasattr(base, "out_features"):
                module.out_features = base.out_features


MODEL_TO_ARCH_MAPPING = {
    "VisionTransformer": "VisionTransformer",
    "DinoVisionTransformer": "DinoVisionTransformer",
    "DINOv3ViTModel": "DINOv3ViTModel",
    "ViTModel": "ViTModel",
    "ViTMAEModel": "ViTModel",
    "Eva": "Eva",
}

MODULE_MAPPING = {
    "VisionTransformer": {
        "attn.proj":  "attn.proj",
        "attn.q_proj": "attn.qkv",  # only exists as qkv, q/k/v usually fused in timm
        "attn.k_proj": "attn.qkv",  # only exists as qkv, q/k/v usually fused in timm
        "attn.v_proj": "attn.qkv",  # only exists as qkv, q/k/v usually fused in timm
        "mlp.fc1":    "mlp.fc1",
        "mlp.fc1_g":  "mlp.fc1_g",  # exists only in gated variants
        "mlp.fc1_x":  "mlp.fc1_x",  # exists only in gated variants
        "mlp.fc2":    "mlp.fc2",
    },
    "DinoVisionTransformer": {
        "attn.proj":  "attn.proj",
        "attn.q_proj": "attn.qkv",
        "attn.k_proj": "attn.qkv",
        "attn.v_proj": "attn.qkv",
        "mlp.fc1":    "mlp.fc1",
        "mlp.fc1_g":  "mlp.fc1_g",
        "mlp.fc1_x":  "mlp.fc1_x",
        "mlp.fc2":    "mlp.fc2",
    },
    "DINOv3ViTModel": {
        "attn.proj":  "attention.o_proj",
        "attn.q_proj": "attention.q_proj",
        "attn.k_proj": "attention.k_proj",
        "attn.v_proj": "attention.v_proj",
        "mlp.fc1":    "mlp.up_proj",
        "mlp.fc1_g":  "mlp.up_proj",
        "mlp.fc1_x":  "mlp.up_proj",
        "mlp.fc2":    "mlp.down_proj",
    },
    "ViTModel": {
        "attn.proj":  "attention.output.dense",
        "attn.q_proj": "attention.attention.query",
        "attn.k_proj": "attention.attention.key",
        "attn.v_proj": "attention.attention.value",
        "mlp.fc1":    "intermediate.dense",
        "mlp.fc1_g":  "intermediate.dense",
        "mlp.fc1_x":  "intermediate.dense",
        "mlp.fc2":    "output.dense",
    },
    "Eva": {
        "attn.proj":  "attn.proj",
        "attn.q_proj": "attn.q_proj",
        "attn.k_proj": "attn.k_proj",
        "attn.v_proj": "attn.v_proj",
        "mlp.fc1":    "mlp.fc1",
        "mlp.fc1_g":  "mlp.fc1_g",
        "mlp.fc1_x":  "mlp.fc1_x",
        "mlp.fc2":    "mlp.fc2",
    },
}


class LoKr:
    """
    LoKr — Low-rank Kronecker Adaptation (LyCORIS / ICLR 2024).

    Decomposes the weight update using a Kronecker product:

        ΔW = W₁ ⊗ W₂

    where W₁ and W₂ are small factor matrices. The Kronecker product
    produces a structured high-rank matrix from two small matrices,
    making it especially efficient for large square weight matrices
    (e.g. 768×768 attention projections) where a full Kronecker product
    covers the entire matrix with fewer parameters than LoHA.

    Complementary to LoHa: LoKr uses multiplicative (Kronecker) structure;
    LoHa uses element-wise (Hadamard) structure.

    Paper: https://arxiv.org/abs/2309.14859
    """

    def __init__(
        self,
        lokr_rank,
        lokr_alpha,
        lokr_dropout,
        lokr_rank_dropout,
        lokr_decompose_factor,
        lokr_target_modules,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        target_arch = MODEL_TO_ARCH_MAPPING[self.model.__class__.__name__]
        module_mapping = MODULE_MAPPING[target_arch]
        lokr_target_modules = list(dict.fromkeys(
            module_mapping[m] for m in lokr_target_modules if m in module_mapping
        ))

        lokr_config = LoKrConfig(
            r=lokr_rank,
            alpha=lokr_alpha,
            module_dropout=lokr_dropout,
            rank_dropout=lokr_rank_dropout,
            decompose_factor=lokr_decompose_factor,
            target_modules=lokr_target_modules,
        )

        self.model = get_peft_model(self.model, lokr_config)
        _patch_peft_linear_compat(self.model)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head", "lokr_"]):
                param.requires_grad = True
