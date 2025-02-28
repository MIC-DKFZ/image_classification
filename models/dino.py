import torch
from base_model import BaseModel
from models.classification_head import ClassificationHead


class DinoV2Model(BaseModel):
    def __init__(self, type, **hypparams):
        super(DinoV2Model, self).__init__(**hypparams)

        size_lookup = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}
        for k in size_lookup.keys():
            if k in type:
                embed_dim = size_lookup[k]

        self.dinov2_encoder = torch.hub.load("facebookresearch/dinov2", type)

        self.cls_head = ClassificationHead(
            embed_dim,
            hypparams["num_classes"],
            dropout=hypparams["classification_head_dropout"],
            patch_aggregation_method=hypparams["token_aggregation_method"],
            cls_token_available=hypparams["cls_token_available"],
        )

    def forward(self, x):

        features = self.dinov2_encoder.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        x = torch.concat([cls_token.unsqueeze(1), patch_tokens], dim=1)

        return self.cls_head(x)
