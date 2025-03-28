import torch
from base_model import BaseModel
from models.classification_head import ClassificationHead
from peft import get_peft_model, LoraConfig


class DinoV2Model(BaseModel):
    def __init__(self, type, **kwargs):
        super(DinoV2Model, self).__init__(**kwargs)

        size_lookup = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}
        for k in size_lookup.keys():
            if k in type:
                embed_dim = size_lookup[k]

        self.encoder = torch.hub.load("facebookresearch/dinov2", type)
        # The mask token is not needed for fine-tuning
        del self.encoder.mask_token

        self.cls_head = ClassificationHead(
            embed_dim,
            kwargs["num_classes"],
            dropout=kwargs["classification_head_dropout"],
            patch_aggregation_method=kwargs["token_aggregation_method"],
        )
        
        if "full" in self.finetune_method:
            pass
            
        elif self.finetune_method == "linear_probing":
            # fully freeze encoder
            self.encoder.requires_grad_(False)
            
        elif self.finetune_method in ["lora", "dora"]:
            # Apply LoRA to attention layers
            lora_config = LoraConfig(
                r=kwargs["lora_rank"],  # LoRA rank
                lora_alpha=kwargs["lora_alpha"],  # Scaling factor
                lora_dropout=kwargs["lora_dropout"],
                target_modules=["attn.proj", "attn.q_proj", "attn.v_proj", "attn.k_proj"],
                use_dora=self.finetune_method == "dora",
            )

            self.encoder = get_peft_model(self.encoder, lora_config)

            # Freeze all layers except LoRA-adapted ones
            self.encoder.requires_grad_(False)
            for name, param in self.encoder.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

        else:
            raise NotImplementedError
    
    @property
    def encoder_params(self):
        return self.encoder.parameters()
    
    @property
    def cls_head_params(self):
        return self.cls_head.parameters()
    
    def on_save_checkpoint(self, checkpoint):
        if self.finetune_method == "linear_probing":
            # Modify checkpoint to only contain classifier weights
            head_state_dict = {
                k: v for k, v in checkpoint["state_dict"].items() if "cls_head" in k
            }
            checkpoint["state_dict"] = head_state_dict

    def forward(self, x):
        features = self.encoder.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        x = torch.concat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        return self.cls_head(x)
    
    def extract_features(self, x):
        features = self.encoder.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        return torch.concat([cls_token.unsqueeze(1), patch_tokens], dim=1)
