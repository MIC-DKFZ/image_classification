from transformers import AutoModel
from base_model import BaseModel
from models.classification_head import ClassificationHead
import torch


class TransformerModel(BaseModel):
    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)

        self.model = AutoModel.from_pretrained(type)  # TODO: Adaptive number of input channels

        if hasattr(self.model, "norm"):
            embed_dim = self.model.norm.normalized_shape[0]
        elif hasattr(self.model, "layernorm"):
            embed_dim = self.model.layernorm.normalized_shape[0]
        else:
            raise RuntimeError("Could not determine embedding dimension for classification head.")

        self.cls_head = ClassificationHead(
            embed_dim,
            kwargs["num_classes"],
            dropout=kwargs["classification_head_dropout"],
            patch_aggregation_method=kwargs["token_aggregation_method"],
        )  
        
        if kwargs.get("classification_head_dropout", None) is not None:
            if hasattr(self.model, "classifier_drop") and isinstance(self.model.classifier_drop, torch.nn.Dropout):
                self.model.classifier_drop.p = kwargs["classification_head_dropout"]
            if hasattr(self.model, "classifier"):
                for name, module in self.model.classifier.named_children():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = kwargs["classification_head_dropout"]
    
    @property
    def encoder_params(self):
        return [
            param for name, param in self.model.named_parameters() if "classifier" not in name
        ]
    
    @property
    def cls_head_params(self):
        return [param for name, param in self.model.named_parameters() if "classifier" in name]

    def forward(self, x):
        x = self.model(x)
        x = x.last_hidden_state
        x = self.cls_head(x)
        return x
