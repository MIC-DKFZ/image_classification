from __future__ import annotations

import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        type: str,
        pretrained: bool,
        input_channels: int,
        model_kwargs: dict | None = None,
    ):
        super().__init__()
        _ = input_channels  # reserved for future multi-channel patch embedding support
        model_kwargs = dict(model_kwargs or {})
        if pretrained:
            self.model = AutoModel.from_pretrained(type, **model_kwargs)
        else:
            cfg = AutoConfig.from_pretrained(type, **model_kwargs)
            self.model = AutoModel.from_config(cfg)

        self.output_dim = self._infer_embed_dim()
        self.features_are_tokens = True

    def _infer_embed_dim(self) -> int:
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)
        if hasattr(self.model, "norm"):
            return int(self.model.norm.normalized_shape[0])
        if hasattr(self.model, "layernorm"):
            return int(self.model.layernorm.normalized_shape[0])
        raise RuntimeError(
            "Could not infer embedding dimension from transformer encoder."
        )

    def forward_features(self, x):
        return self.model(x).last_hidden_state
