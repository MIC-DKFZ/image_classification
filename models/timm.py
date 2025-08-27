import timm
import torch
from base_model import BaseModel


class TimmModel(BaseModel):
    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)

        self.model = timm.create_model(
            type,
            pretrained=kwargs["pretrained"],
            in_chans=kwargs["input_channels"],
            num_classes=kwargs["num_classes"],
        ) 
        
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
