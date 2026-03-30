from peft import get_peft_model, LoraConfig


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
        "attn.proj": "attn.proj",
        "attn.q_proj": "attn.qkv",  # only exists as qkv, q/k/v usually fused in timm
        "attn.k_proj": "attn.qkv",  # only exists as qkv, q/k/v usually fused in timm
        "attn.v_proj": "attn.qkv",  # only exists as qkv, q/k/v usually fused in timm
        "mlp.fc1": "mlp.fc1",
        "mlp.fc1_g": "mlp.fc1_g",  # exists only in gated variants
        "mlp.fc1_x": "mlp.fc1_x",  # exists only in gated variants
        "mlp.fc2":   "mlp.fc2",
    },
    "DinoVisionTransformer": {
        "attn.proj": "attn.proj",
        "attn.q_proj": "attn.qkv",
        "attn.k_proj": "attn.qkv",
        "attn.v_proj": "attn.qkv",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc1_g": "mlp.fc1_g",
        "mlp.fc1_x": "mlp.fc1_x",
        "mlp.fc2":   "mlp.fc2",
    },
    "DINOv3ViTModel": {
        "attn.proj": "attention.o_proj",
        "attn.q_proj": "attention.q_proj",
        "attn.k_proj": "attention.k_proj",
        "attn.v_proj": "attention.v_proj",
        "mlp.fc1": "mlp.up_proj",
        "mlp.fc1_g": "mlp.up_proj",
        "mlp.fc1_x": "mlp.up_proj",  # MLP is not gated
        "mlp.fc2":   "mlp.down_proj",
    },
    "ViTModel": {
        "attn.proj":  "attention.output.dense",
        "attn.q_proj":"attention.attention.query",
        "attn.k_proj":"attention.attention.key",
        "attn.v_proj":"attention.attention.value",
        "mlp.fc1": "intermediate.dense",
        "mlp.fc1_g": "intermediate.dense",
        "mlp.fc1_x": "intermediate.dense",  # MLP is not gated
        "mlp.fc2":   "output.dense",
    },
    "Eva": {
        "attn.proj":  "attn.proj",
        "attn.q_proj":"attn.q_proj",
        "attn.k_proj":"attn.k_proj",
        "attn.v_proj":"attn.v_proj",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc1_g": "mlp.fc1_g",
        "mlp.fc1_x": "mlp.fc1_x",
        "mlp.fc2":   "mlp.fc2",
    },
}


class LoRA:
    def __init__(self, lora_rank, lora_alpha, lora_dropout, lora_target_modules, use_dora,
                 use_rslora, init_lora_weights, lora_bias, *args, **kwargs):
        super().__init__(*args, **kwargs)

        target_arch = MODEL_TO_ARCH_MAPPING[self.model.__class__.__name__]
        module_mapping = MODULE_MAPPING[target_arch]
        lora_target_modules = [module_mapping[module] for module in lora_target_modules]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            use_dora=use_dora,
            use_rslora=use_rslora,
            init_lora_weights=init_lora_weights,
            bias=lora_bias,
        )

        self.model = get_peft_model(self.model, lora_config)

        # Freeze all layers except LoRA-adapted ones
        for param in self.model.parameters():
            param.requires_grad = False

        unfreeze_subs = ["head", "classifier", "cls_head", "lora"]
        if lora_bias == "all":
            unfreeze_subs.append(".bias")
        elif lora_bias == "lora_only":
            unfreeze_subs.append("base_layer.bias")

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in unfreeze_subs):
                param.requires_grad = True

