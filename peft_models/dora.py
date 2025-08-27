from peft import get_peft_model, LoraConfig


class DoRA:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lora_config = LoraConfig(
            r=kwargs['lora_rank'],  # LoRA rank
            lora_alpha=kwargs['lora_alpha'],  # Scaling factor
            lora_dropout=kwargs['lora_dropout'],
            target_modules=["attn.proj", "attn.q_proj", "attn.v_proj", "attn.k_proj", "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2"],
            use_dora=True,
        )

        self.model = get_peft_model(self.model, lora_config)

        # Freeze all layers except LoRA-adapted ones
        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            if "head" in name:
                param.requires_grad = True

