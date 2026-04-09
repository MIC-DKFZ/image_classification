from __future__ import annotations

import torch.nn as nn

from glovita.models.video_encoder.common import (
    capture_named_intermediates,
    default_intermediate_names,
    replace_first_conv3d,
    strip_last_linear,
)


class PytorchvideoEncoder(nn.Module):
    def __init__(
        self,
        type: str,
        pretrained: bool,
        input_channels: int,
        pathway_mode: str = "auto",
        slowfast_alpha: int = 4,
        return_intermediates: bool = False,
        intermediate_names: list[str] | None = None,
        model_kwargs: dict | None = None,
    ):
        super().__init__()
        import pytorchvideo.models.hub as pvhub

        build_fn = getattr(pvhub, type)
        build_kwargs = dict(model_kwargs or {})
        build_kwargs["pretrained"] = pretrained
        self.model = build_fn(**build_kwargs)
        if input_channels != 3 and not replace_first_conv3d(self.model, input_channels):
            raise ValueError(
                f"pytorchvideo model {type!r} does not support automatic input channel patching."
            )
        self.output_dim = strip_last_linear(self.model)
        self.features_are_tokens = False
        self.pathway_mode = self._resolve_pathway_mode(type, pathway_mode)
        self.slowfast_alpha = int(slowfast_alpha)
        self.return_intermediates = bool(return_intermediates)
        self.intermediate_names = list(intermediate_names or [])
        if self.return_intermediates and not self.intermediate_names:
            self.intermediate_names = default_intermediate_names(self.model)

    @staticmethod
    def _resolve_pathway_mode(model_type: str, pathway_mode: str) -> str:
        if pathway_mode == "auto":
            return "slowfast" if "slowfast" in model_type.lower() else "single"
        return pathway_mode

    def _prepare_input(self, x):
        if self.pathway_mode == "single":
            return x
        if self.pathway_mode == "slowfast":
            return [x[:, :, :: self.slowfast_alpha], x]
        raise ValueError(f"Unknown pytorchvideo pathway_mode={self.pathway_mode!r}.")

    def forward_features(self, x):
        prepared_x = self._prepare_input(x)
        if not self.return_intermediates:
            return self.model(prepared_x)
        features, intermediates = capture_named_intermediates(
            self.model,
            self.intermediate_names,
            lambda: self.model(prepared_x),
        )
        return {"features": features, "intermediates": intermediates}
