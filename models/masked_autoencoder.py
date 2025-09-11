import torch
from base_model import BaseModel
from models.classification_head import ClassificationHead
from functools import partial
import torch.nn as nn
import timm.models.vision_transformer
from timm.models.vision_transformer import VisionTransformer


class MaskedAutoencoder(BaseModel):
    def __init__(self, type, weight_dir, **kwargs):
        super().__init__(**kwargs)

        self.model = globals()[type]()
        if hasattr(self.model, "head"):
            del self.model.head

        state_dict = torch.load(weight_dir, map_location='cpu')["model"]
        self.model.load_state_dict(state_dict, strict=False)

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
        x = self.model.forward_features(x)
        x = self.cls_head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    return model