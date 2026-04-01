from peft import get_peft_model, BOFTConfig


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


class BOFT:
    """
    BOFT — Butterfly Orthogonal Fine-Tuning (ICLR 2024).

    Extends OFT by factorising the orthogonal transformation using butterfly
    matrices, dramatically reducing parameter count while preserving the
    geometric structure-preservation property of OFT.

    The orthogonal matrix is expressed as a product of m butterfly factors,
    each operating on non-overlapping blocks of size boft_block_size.
    boft_n_butterfly_factor controls the depth (m) of this factorisation:
    higher m → more expressive but more parameters.

    Prefer boft_block_size over boft_block_num; only set one of the two.

    Paper: https://arxiv.org/abs/2311.06243
    """

    def __init__(
        self,
        boft_block_size,        # size of each butterfly block (e.g. 8)
        boft_n_butterfly_factor,  # depth of butterfly factorisation (e.g. 1 or 2)
        boft_dropout,
        boft_target_modules,
        boft_bias,
        *args, **kwargs,
    ):
        target_arch = MODEL_TO_ARCH_MAPPING[self.model.__class__.__name__]
        module_mapping = MODULE_MAPPING[target_arch]
        boft_target_modules = list(dict.fromkeys(
            module_mapping[m] for m in boft_target_modules
        ))

        boft_config = BOFTConfig(
            boft_block_size=boft_block_size,
            boft_n_butterfly_factor=boft_n_butterfly_factor,
            boft_dropout=boft_dropout,
            target_modules=boft_target_modules,
            bias=boft_bias,
        )

        self.model = get_peft_model(self.model, boft_config)

        for param in self.model.parameters():
            param.requires_grad = False

        unfreeze_subs = ["head", "classifier", "cls_head", "boft_"]
        if boft_bias == "all":
            unfreeze_subs.append(".bias")
        elif boft_bias == "boft_only":
            unfreeze_subs.append("base_layer.bias")

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in unfreeze_subs):
                param.requires_grad = True
