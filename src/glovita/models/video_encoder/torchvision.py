from __future__ import annotations

import torch.nn as nn

from glovita.models.video_encoder.common import (
    capture_named_intermediates,
    default_intermediate_names,
    filter_supported_kwargs,
    replace_first_conv3d,
    strip_last_linear,
)


class TorchvisionVideoEncoder(nn.Module):
    def __init__(
        self,
        type: str,
        pretrained: bool,
        input_channels: int,
        return_intermediates: bool = False,
        intermediate_names: list[str] | None = None,
        model_kwargs: dict | None = None,
    ):
        super().__init__()
        import torchvision.models.video as tvv

        build_fn = getattr(tvv, type)
        build_kwargs = dict(model_kwargs or {})
        if pretrained:
            weights = tvv.get_model_weights(type).DEFAULT
            build_kwargs["weights"] = weights
        else:
            build_kwargs["weights"] = None
        self.model = build_fn(**filter_supported_kwargs(build_fn, build_kwargs))
        if input_channels != 3 and not replace_first_conv3d(self.model, input_channels):
            raise ValueError(
                f"torchvision video model {type!r} does not support automatic input channel patching."
            )
        self.output_dim = strip_last_linear(self.model)
        self.features_are_tokens = False
        self.return_intermediates = bool(return_intermediates)
        self.intermediate_names = list(intermediate_names or [])
        if self.return_intermediates and not self.intermediate_names:
            self.intermediate_names = default_intermediate_names(self.model)

    def forward_features(self, x):
        if not self.return_intermediates:
            return self.model(x)
        features, intermediates = capture_named_intermediates(
            self.model,
            self.intermediate_names,
            lambda: self.model(x),
        )
        return {"features": features, "intermediates": intermediates}
