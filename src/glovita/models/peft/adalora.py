from peft import get_peft_model, AdaLoraConfig


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


class AdaLoRA:
    """
    AdaLoRA — Adaptive Low-Rank Adaptation (ICLR 2023).

    Extends LoRA by parameterising weight updates in SVD form (P·Λ·Q) and
    dynamically pruning singular values based on importance scores, so rank
    budget is redistributed across layers during training rather than fixed.

    Training note: the training loop must call
        model.base_model.update_and_allocate(global_step)
    at each optimiser step so importance scores are updated and rank is reallocated.

    Paper: https://arxiv.org/abs/2303.10512
    """

    def __init__(
        self,
        adalora_rank,         # target rank after pruning
        adalora_init_rank,    # initial rank before pruning (should be >= adalora_rank)
        adalora_alpha,
        adalora_dropout,
        adalora_target_modules,
        use_rslora,
        init_lora_weights,
        lora_bias,
        adalora_orth_reg_weight,
        adalora_beta1,
        adalora_beta2,
        adalora_tinit,
        adalora_deltaT,
        *args, **kwargs,
    ):
        target_arch = MODEL_TO_ARCH_MAPPING[self.model.__class__.__name__]
        module_mapping = MODULE_MAPPING[target_arch]
        adalora_target_modules = list(dict.fromkeys(
            module_mapping[m] for m in adalora_target_modules
        ))

        adalora_config = AdaLoraConfig(
            target_r=adalora_rank,
            init_r=adalora_init_rank,
            lora_alpha=adalora_alpha,
            lora_dropout=adalora_dropout,
            target_modules=adalora_target_modules,
            use_rslora=use_rslora,
            init_lora_weights=init_lora_weights,
            bias=lora_bias,
            orth_reg_weight=adalora_orth_reg_weight,
            beta1=adalora_beta1,
            beta2=adalora_beta2,
            tinit=adalora_tinit,
            deltaT=adalora_deltaT,
        )

        self.model = get_peft_model(self.model, adalora_config)

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

    def on_train_start(self):
        total_steps = int(self.trainer.estimated_stepping_batches)
        for config in self.model.peft_config.values():
            config.total_step = total_steps

    def on_before_optimizer_step(self, optimizer):
        if hasattr(self.model, "base_model"):
            import torch
            # kthvalue has no deterministic CUDA impl; disable briefly if needed
            _det = torch.are_deterministic_algorithms_enabled()
            if _det:
                torch.use_deterministic_algorithms(False)
            try:
                self.model.base_model.update_and_allocate(self.global_step)
            finally:
                if _det:
                    torch.use_deterministic_algorithms(True)
