import torch
from base_model import BaseModel
from models.classification_head import ClassificationHead
from pathlib import Path


class Dinov3(BaseModel):
    def __init__(self, type, weight_dir, **kwargs):
        super().__init__(**kwargs)

        model_name = str("_").join(type.split("_")[:2])
        self.model = torch.hub.load("facebookresearch/dinov3", model_name, weights=str(Path(weight_dir) / (type + ".pth")))
        embed_dim = self.model.embed_dim

        self.cls_head = ClassificationHead(
            embed_dim,
            kwargs["num_classes"],
            dropout=kwargs["classification_head_dropout"],
            patch_aggregation_method=kwargs["token_aggregation_method"],
        )
    
    @property
    def encoder_params(self):
        return self.model.parameters()
    
    @property
    def cls_head_params(self):
        return self.cls_head.parameters()

    def forward(self, x):
        features = self.model.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        x = torch.concat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        return self.cls_head(x)
    
    def extract_features(self, x):
        features = self.model.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        return torch.concat([cls_token.unsqueeze(1), patch_tokens], dim=1)
