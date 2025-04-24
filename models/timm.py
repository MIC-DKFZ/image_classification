import timm
import torch
from base_model import BaseModel
from peft import get_peft_model, LoraConfig, TaskType


class TimmModel(BaseModel):
    def __init__(self, type, **kwargs):
        super(TimmModel, self).__init__(**kwargs)

        self.model = timm.create_model(
            type,
            pretrained=kwargs["pretrained"],
            in_chans=kwargs["input_channels"],
            num_classes=kwargs["num_classes"],
        )

        if "full" in kwargs["finetune_method"]:
            pass
            
        elif kwargs["finetune_method"] == "linear_probing":
            # fully freeze encoder
            for name, param in self.model.named_parameters():
                if "head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
        elif kwargs["finetune_method"] == "lora":
            # Apply LoRA to attention layers

            lora_config = LoraConfig(
                r=kwargs['lora_rank'],  # LoRA rank
                lora_alpha=kwargs['lora_alpha'],  # Scaling factor
                lora_dropout=kwargs['lora_dropout'],
                target_modules=["attn.proj", "attn.q_proj", "attn.v_proj", "attn.k_proj", "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2"]
            )

            self.model = get_peft_model(self.model, lora_config)

            # Freeze all layers except LoRA-adapted ones
            for param in self.model.parameters():
                param.requires_grad = False

            for name, param in self.model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

        elif kwargs["finetune_method"] == "dora":
            # Apply LoRA to attention layers

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

        else:
            raise NotImplementedError
        
        if kwargs.get("classification_head_dropout", None) is not None:
            if hasattr(self.model, "head_drop") and isinstance(self.model.head_drop, torch.nn.Dropout):
                self.model.head_drop.p = kwargs["classification_head_dropout"]
            if hasattr(self.model, "head"):
                for name, module in self.model.head.named_children():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = kwargs["classification_head_dropout"]
    
    @property
    def encoder_params(self):
        return [
            param for name, param in self.model.named_parameters() if "head" not in name
        ]
    
    @property
    def cls_head_params(self):
        return [param for name, param in self.model.named_parameters() if "head" in name]

    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        # This works for most models in timm
        return self.model.forward_features(x)
