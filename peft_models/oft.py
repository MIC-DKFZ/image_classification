from peft import get_peft_model, OFTConfig


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


class OFT:
    """
    OFT — Orthogonal Fine-Tuning (NeurIPS 2023).

    Adapts pre-trained weights by multiplying with a block-diagonal orthogonal
    matrix R: W' = R · W₀. Because R is orthogonal, the pairwise angles
    (hyperspherical energy) between weight vectors are preserved, keeping the
    pre-trained feature geometry intact while allowing task-specific rotation.

    r controls the number of blocks; block size = out_features / r.
    Larger r → smaller blocks → fewer parameters but less expressive.

    Paper: https://arxiv.org/abs/2306.07280
    """

    def __init__(
        self,
        oft_r,                 # number of OFT blocks (block size = out_dim / r)
        oft_dropout,
        oft_target_modules,
        oft_coft,
        oft_eps,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        target_arch = MODEL_TO_ARCH_MAPPING[self.model.__class__.__name__]
        module_mapping = MODULE_MAPPING[target_arch]
        oft_target_modules = list(dict.fromkeys(
            module_mapping[m] for m in oft_target_modules
        ))

        oft_config = OFTConfig(
            r=oft_r,
            module_dropout=oft_dropout,
            target_modules=oft_target_modules,
            coft=oft_coft,
            eps=oft_eps,
        )

        self.model = get_peft_model(self.model, oft_config)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head", "oft_"]):
                param.requires_grad = True
