import torch
from base_model import BaseModel
from models.classification_head import ClassificationHead
from timm.layers import ClassifierHead


class LinearModel(BaseModel):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        
        if kwargs["token_aggregation_method"] == "joint":
            embed_dim *= 2

        # directly apply the FC layer on already aggregated (precomputed) features
        if kwargs.get("aggregated", False):
            self.model = ClassifierHead(
                embed_dim,
                num_classes=kwargs["num_classes"],
                pool_type="",
                drop_rate=kwargs["classification_head_dropout"],
            )
        # or employ ClassificationHead which applies the token aggregation method first
        else:
            self.model = ClassificationHead(
                embed_dim,
                kwargs["num_classes"],
                dropout=kwargs["classification_head_dropout"],
                patch_aggregation_method=kwargs["token_aggregation_method"],
            )

    def forward(self, x):
        return self.model(x)
