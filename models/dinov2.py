import torch
from base_model import BaseModel
from models.classification_head import ClassificationHead


class Dinov2(BaseModel):
    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)

        size_lookup = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}
        for k in size_lookup.keys():
            if k in type:
                embed_dim = size_lookup[k]

        self.model = torch.hub.load("facebookresearch/dinov2", type)
        # The mask token is not needed for fine-tuning
        del self.model.mask_token

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
