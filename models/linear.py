import torch
from base_model import BaseModel
from models.classification_head import ClassificationHead


class LinearModel(BaseModel):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.model = ClassificationHead(
            embed_dim,
            kwargs["num_classes"],
            dropout=kwargs["classification_head_dropout"],
            patch_aggregation_method=kwargs["token_aggregation_method"],
        )

    def forward(self, x):
        return self.model(x)
