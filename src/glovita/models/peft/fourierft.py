from peft import get_peft_model, FourierFTConfig


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


class FourierFT:
    """
    FourierFT — Fourier-domain Parameter-Efficient Fine-Tuning (NeurIPS 2024).

    Instead of learning ΔW directly or as a low-rank product, FourierFT
    learns the weight update in the **discrete Fourier frequency domain**:

        ΔW = IDFT( sparse_spectrum )

    Only `n_frequency` frequency components (out of d_out × d_in total) are
    learned — a sparse selection in frequency space. The inverse DFT then
    reconstructs the full-rank ΔW at each forward pass.

    This produces a qualitatively different type of update from LoRA-family
    methods: low-rank methods capture axis-aligned structure in weight space;
    FourierFT captures oscillatory / frequency-domain structure. With very
    small n_frequency (e.g. 100–1000), the parameter count is a fixed small
    number independent of the weight matrix size, making it competitive with
    VeRA and FacT at the extreme low-parameter end.

    Paper: https://arxiv.org/abs/2405.03003
    """

    def __init__(
        self,
        fourierft_n_frequency,
        fourierft_scaling,
        fourierft_target_modules,
        *args, **kwargs,
    ):
        target_arch = MODEL_TO_ARCH_MAPPING[self.model.__class__.__name__]
        module_mapping = MODULE_MAPPING[target_arch]
        fourierft_target_modules = list(dict.fromkeys(
            module_mapping[m] for m in fourierft_target_modules
        ))

        fourierft_config = FourierFTConfig(
            n_frequency=fourierft_n_frequency,
            scaling=fourierft_scaling,
            target_modules=fourierft_target_modules,
        )

        self.model = get_peft_model(self.model, fourierft_config)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head", "spectrum"]):
                param.requires_grad = True
