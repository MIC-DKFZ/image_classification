from peft import get_peft_model, IA3Config


MODEL_TO_ARCH_MAPPING = {
    "VisionTransformer": "VisionTransformer",
    "DinoVisionTransformer": "DinoVisionTransformer",
    "DINOv3ViTModel": "DINOv3ViTModel",
    "ViTModel": "ViTModel",
    "ViTMAEModel": "ViTModel",
    "Eva": "Eva",
}

# attn_modules: K and V projections (or fused QKV for timm)
# ff_modules:   first FFN linear (pre-activation); must be subset of attn_modules + ff_modules
IA3_TARGETS = {
    "VisionTransformer": {
        "attn_modules": ["attn.qkv"],       # fused Q,K,V — scales the full 3D output
        "ff_modules":   ["mlp.fc1"],
    },
    "DinoVisionTransformer": {
        "attn_modules": ["attn.qkv"],       # fused Q,K,V — scales the full 3D output
        "ff_modules":   ["mlp.fc1"],
    },
    "DINOv3ViTModel": {
        "attn_modules": ["attention.k_proj", "attention.v_proj"],
        "ff_modules":   ["mlp.up_proj"],
    },
    "ViTModel": {
        "attn_modules": ["attention.attention.key", "attention.attention.value"],
        "ff_modules":   ["intermediate.dense"],
    },
    "Eva": {
        "attn_modules": ["attn.k_proj", "attn.v_proj"],  # Eva has separate q/k/v
        "ff_modules":   ["mlp.fc1"],
    },
}


class IA3:
    """
    IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations (NeurIPS 2022).

    Learns one scaling vector l per targeted linear layer; the forward pass becomes:
        output = (l ⊙ W·x)
    Applied to K, V projections in attention and the pre-activation FFN output.
    Typically <0.1% of ViT-B parameters.

    Paper: https://arxiv.org/abs/2205.05638
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        target_arch = MODEL_TO_ARCH_MAPPING[self.model.__class__.__name__]
        targets = IA3_TARGETS[target_arch]

        target_modules = list(dict.fromkeys(targets["attn_modules"] + targets["ff_modules"]))

        ia3_config = IA3Config(
            target_modules=target_modules,
            feedforward_modules=targets["ff_modules"],
        )

        self.model = get_peft_model(self.model, ia3_config)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head", "ia3"]):
                param.requires_grad = True
