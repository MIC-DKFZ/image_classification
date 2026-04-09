from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def filter_supported_kwargs(fn, kwargs: dict) -> dict:
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    accepted = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in accepted}


def replace_first_conv3d(module: nn.Module, input_channels: int) -> bool:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv3d):
            if child.in_channels != 3:
                return False
            new_conv = nn.Conv3d(
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
        if replace_first_conv3d(child, input_channels):
            return True
    return False


def infer_linear_dim(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if isinstance(module, nn.Sequential):
        for child in reversed(list(module.children())):
            dim = infer_linear_dim(child)
            if dim is not None:
                return dim
    if isinstance(module, nn.ModuleList):
        for child in reversed(list(module)):
            dim = infer_linear_dim(child)
            if dim is not None:
                return dim
    for child in reversed(list(module.children())):
        dim = infer_linear_dim(child)
        if dim is not None:
            return dim
    return None


def _set_module_by_path(root: nn.Module, dotted_path: str, module: nn.Module) -> None:
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def strip_last_linear(backbone: nn.Module) -> int:
    last_path = None
    last_dim = None
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Linear):
            last_path = name
            last_dim = int(module.in_features)
    if last_path is None or last_dim is None:
        raise RuntimeError(
            f"Could not find a linear classifier head on {backbone.__class__.__name__}."
        )
    _set_module_by_path(backbone, last_path, nn.Identity())
    return last_dim


def default_intermediate_names(model: nn.Module) -> list[str]:
    if all(hasattr(model, name) for name in ("stem", "layer1", "layer2", "layer3", "layer4")):
        return ["stem", "layer1", "layer2", "layer3", "layer4"]
    return []


def capture_named_intermediates(
    model: nn.Module,
    module_names: list[str],
    forward_fn,
):
    outputs: dict[str, torch.Tensor] = {}
    module_map = dict(model.named_modules())
    missing = [name for name in module_names if name not in module_map]
    if missing:
        raise ValueError(
            f"Requested intermediate modules are not present on {model.__class__.__name__}: {missing}"
        )

    hooks = []
    for name in module_names:
        module = module_map[name]

        def _make_hook(module_name: str):
            def _hook(_module, _inputs, output):
                outputs[module_name] = output

            return _hook

        hooks.append(module.register_forward_hook(_make_hook(name)))

    try:
        final_output = forward_fn()
    finally:
        for hook in hooks:
            hook.remove()

    return final_output, outputs
