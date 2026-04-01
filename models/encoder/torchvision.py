from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def _filter_supported_kwargs(fn, kwargs: dict) -> dict:
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    accepted = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in accepted}


def _replace_first_conv(module: nn.Module, input_channels: int) -> bool:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.in_channels != 3:
                return False
            new_conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            with torch.no_grad():
                if input_channels == 1:
                    new_conv.weight.copy_(child.weight.mean(dim=1, keepdim=True))
                elif input_channels > 3:
                    new_conv.weight.zero_()
                    new_conv.weight[:, :3].copy_(child.weight)
                else:
                    new_conv.weight[:, :input_channels].copy_(child.weight[:, :input_channels])
                if child.bias is not None:
                    new_conv.bias.copy_(child.bias)
            setattr(module, name, new_conv)
            return True
        if _replace_first_conv(child, input_channels):
            return True
    return False


def _infer_linear_dim(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if isinstance(module, nn.Sequential):
        for child in reversed(list(module.children())):
            dim = _infer_linear_dim(child)
            if dim is not None:
                return dim
    return None


def _strip_classifier(backbone: nn.Module) -> tuple[int, bool]:
    for attr in ("fc", "classifier", "head", "heads"):
        if not hasattr(backbone, attr):
            continue
        module = getattr(backbone, attr)
        dim = _infer_linear_dim(module)
        if dim is None:
            continue
        setattr(backbone, attr, nn.Identity())
        return dim, False
    raise RuntimeError(
        f"Could not strip classifier from torchvision model {backbone.__class__.__name__}."
    )


class TorchvisionEncoder(nn.Module):
    def __init__(
        self,
        type: str,
        pretrained: bool,
        input_channels: int,
        dropout: float | None = None,
        stochastic_depth_prob: float | None = None,
    ):
        super().__init__()
        import torchvision.models as tvm

        build_fn = getattr(tvm, type)
        build_kwargs = {}
        if pretrained:
            weights_enum = tvm.get_model_weights(type)
            build_kwargs["weights"] = weights_enum.DEFAULT
        else:
            build_kwargs["weights"] = None
        if dropout is not None:
            build_kwargs["dropout"] = dropout
        if stochastic_depth_prob is not None:
            build_kwargs["stochastic_depth_prob"] = stochastic_depth_prob

        self.model = build_fn(**_filter_supported_kwargs(build_fn, build_kwargs))
        if input_channels != 3 and not _replace_first_conv(self.model, input_channels):
            raise ValueError(
                f"torchvision model {type!r} does not support automatic input channel patching."
            )
        self.output_dim, self.features_are_tokens = _strip_classifier(self.model)

    def forward_features(self, x):
        return self.model(x)
