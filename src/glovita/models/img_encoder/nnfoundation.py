"""nnFoundation encoders built from the `nnssl_adaptation_plan` stored in the checkpoint."""
from __future__ import annotations

import inspect
import math
import pkgutil
import pydoc
import warnings
from importlib import import_module
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from glovita.models.img_encoder.dynamic import primus_forward_features


_PRIMUS_MODULE = "dynamic_network_architectures.architectures.primus"
_PRIMUS_CLASS = f"{_PRIMUS_MODULE}.Primus"
_PRIMUS_PRESETS = ("PrimusS", "PrimusB", "PrimusM", "PrimusL")
_PRIMUS_PRESET_PATCH_EMBED_SIZE = (8, 8, 8)

_PRIMUS_KWARGS = (
    "embed_dim",
    "patch_embed_size",
    "eva_depth",
    "eva_numheads",
    "input_shape",
    "use_rot_pos_emb",
    "use_abs_pos_embed",
    "drop_path_rate",
    "init_values",
    "scale_attn_inner",
    "num_register_tokens",
)
_PRIMUS_DEFAULT_KWARGS = {
    "use_rot_pos_emb": True,
    "use_abs_pos_embed": True,
    "drop_path_rate": 0.2,
    "init_values": 0.1,
    "scale_attn_inner": True,
    "num_register_tokens": 0,
}

_ARCHITECTURE_PRESETS: dict[str, tuple[str, dict, list[str]]] = {
    "ResEncL": (
        "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[3, 3, 3]] * 6,
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        ["conv_op", "norm_op", "dropout_op", "nonlin"],
    ),
}


def load_nnfoundation_checkpoint(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)


def get_pretrain_patch_size(plan: dict) -> tuple[int, ...]:
    configurations = plan["pretrain_plan"]["configurations"]
    return tuple(next(iter(configurations.values()))["patch_size"])


def get_architecture_from_plan(plan: dict) -> tuple[str, dict, list[str]]:
    architecture_plans = plan["architecture_plans"]
    arch_class_name = architecture_plans["arch_class_name"]

    if arch_class_name in _ARCHITECTURE_PRESETS:
        class_name, arch_kwargs, req_import = _ARCHITECTURE_PRESETS[arch_class_name]
        return class_name, dict(arch_kwargs), list(req_import)

    if arch_class_name in _PRIMUS_PRESETS:
        arch_kwargs = {
            "patch_embed_size": _PRIMUS_PRESET_PATCH_EMBED_SIZE,
            "input_shape": get_pretrain_patch_size(plan),
            "drop_path_rate": _PRIMUS_DEFAULT_KWARGS["drop_path_rate"],
        }
        return f"{_PRIMUS_MODULE}.{arch_class_name}", arch_kwargs, []

    arch_kwargs = dict(architecture_plans["arch_kwargs"])
    if arch_class_name.split(".")[-1] in ("Primus", "PrimusX"):
        arch_kwargs = {key.removeprefix("encoder_"): value for key, value in arch_kwargs.items()}
        arch_kwargs = {**_PRIMUS_DEFAULT_KWARGS, **arch_kwargs}
        arch_kwargs = {key: value for key, value in arch_kwargs.items() if key in _PRIMUS_KWARGS}
        arch_kwargs["patch_embed_size"] = tuple(arch_kwargs["patch_embed_size"])
        arch_kwargs["input_shape"] = tuple(arch_kwargs["input_shape"])
        return _PRIMUS_CLASS, arch_kwargs, []

    return arch_class_name, arch_kwargs, list(architecture_plans.get("arch_kwargs_requiring_import") or [])


def _find_dynamic_network_architecture(class_name: str):
    import dynamic_network_architectures.architectures as architectures

    for module_info in pkgutil.walk_packages(architectures.__path__, prefix=f"{architectures.__name__}."):
        module = import_module(module_info.name)
        if hasattr(module, class_name):
            return getattr(module, class_name)
    return None


def get_network_from_plans(
    arch_class_name: str,
    arch_kwargs: dict,
    arch_kwargs_req_import: list[str],
    input_channels: int,
    output_channels: int,
    allow_init: bool = True,
    deep_supervision: bool | None = None,
) -> nn.Module:
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(arch_class_name)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        if "." in arch_class_name:
            warnings.warn(
                f"Network class {arch_class_name} not found. Attempting to locate it within "
                "dynamic_network_architectures.architectures..."
            )
        nw_class = _find_dynamic_network_architecture(arch_class_name.split(".")[-1])
        if nw_class is None:
            raise ImportError(f"Network class {arch_class_name} could not be found.")

    if deep_supervision is not None:
        architecture_kwargs["deep_supervision"] = deep_supervision

    output_kwarg = "output_channels" if "output_channels" in inspect.signature(nw_class).parameters else "num_classes"
    network = nw_class(
        input_channels=input_channels,
        **{output_kwarg: output_channels},
        **architecture_kwargs,
    )

    if hasattr(network, "initialize") and allow_init:
        network.apply(network.initialize)

    return network


def resize_pos_embed(
    pos_embed: torch.Tensor, source_grid: tuple[int, ...], target_grid: tuple[int, ...]
) -> torch.Tensor:
    num_prefix_tokens = pos_embed.shape[1] - math.prod(source_grid)
    prefix_tokens, patch_tokens = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
    patch_tokens = patch_tokens.transpose(1, 2).reshape(1, -1, *source_grid)
    patch_tokens = F.interpolate(patch_tokens.float(), size=target_grid, mode="trilinear", align_corners=False)
    patch_tokens = patch_tokens.flatten(2).transpose(1, 2).to(pos_embed.dtype)
    return torch.cat((prefix_tokens, patch_tokens), dim=1)


def load_pretrained_weights(
    network: nn.Module,
    checkpoint: dict,
    input_channels: int,
    pos_embed_grids: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> None:
    plan = checkpoint["nnssl_adaptation_plan"]
    prefixes = (plan["key_to_stem"], plan["key_to_encoder"])
    pretrained_dict = {k: v for k, v in checkpoint["network_weights"].items() if k.startswith(prefixes)}

    key_to_lpe = plan.get("key_to_lpe")
    if key_to_lpe is not None and pos_embed_grids is not None and pos_embed_grids[0] != pos_embed_grids[1]:
        pretrained_dict[key_to_lpe] = resize_pos_embed(pretrained_dict[key_to_lpe], *pos_embed_grids)

    pretrain_input_channels = plan["pretrain_num_input_channels"]
    if input_channels > pretrain_input_channels:
        if pretrain_input_channels != 1:
            raise NotImplementedError(
                "Adapting the input projection is only supported for single-channel pretraining, "
                f"got {pretrain_input_channels} pretraining channels."
            )
        for key in plan["keys_to_in_proj"]:
            weight = pretrained_dict[f"{key}.weight"]
            pretrained_dict[f"{key}.weight"] = (
                weight.repeat(1, input_channels, *([1] * (weight.ndim - 2))) / input_channels
            )

    model_dict = network.state_dict()
    for key in model_dict:
        if key.startswith(prefixes):
            assert key in pretrained_dict, (
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be "
                "compatible with your network."
            )
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}."
            )

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)


class nnFoundationEncoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: Path,
        pretrained: bool,
        input_channels: int,
        drop_path_rate: float | None,
        input_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        from dynamic_network_architectures.architectures.primus import Primus
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

        checkpoint = load_nnfoundation_checkpoint(checkpoint_path)
        plan = checkpoint["nnssl_adaptation_plan"]
        arch_class_name, arch_kwargs, arch_kwargs_req_import = get_architecture_from_plan(plan)
        is_primus = arch_class_name.startswith(f"{_PRIMUS_MODULE}.")
        if drop_path_rate is not None:
            if not is_primus:
                raise ValueError("drop_path_rate is only supported for PRIMUS nnFoundation encoders.")
            arch_kwargs["drop_path_rate"] = drop_path_rate

        pos_embed_grids = None
        if is_primus:
            pretrain_shape = get_pretrain_patch_size(plan)
            target_shape = tuple(input_shape or plan["recommended_downstream_patchsize"])
            arch_kwargs["input_shape"] = target_shape
            patch_embed_size = arch_kwargs["patch_embed_size"]
            pos_embed_grids = tuple(
                tuple(s // p for s, p in zip(shape, patch_embed_size)) for shape in (pretrain_shape, target_shape)
            )

        self.model = get_network_from_plans(
            arch_class_name,
            arch_kwargs,
            arch_kwargs_req_import,
            input_channels=input_channels,
            output_channels=1,
            allow_init=True,
            deep_supervision=None if is_primus else False,
        )
        if pretrained:
            load_pretrained_weights(self.model, checkpoint, input_channels, pos_embed_grids)
        del checkpoint
        self.adaptation_plan = plan

        if isinstance(self.model, Primus):
            self.model.up_projection = nn.Identity()
            self.model.mask_token = None
            self.output_dim = int(self.model.embed_dim)
        elif isinstance(self.model, ResidualEncoderUNet):
            self.model.decoder = nn.Identity()
            self.model.encoder.return_skips = False
            self.output_dim = int(arch_kwargs["features_per_stage"][-1])
        else:
            raise NotImplementedError(f"Unsupported nnFoundation architecture: {type(self.model).__name__}")
        self.features_are_tokens = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        from dynamic_network_architectures.architectures.primus import Primus

        if isinstance(self.model, Primus):
            return primus_forward_features(self.model, x)
        features = self.model.encoder(x)
        return features.mean(dim=tuple(range(2, features.ndim)))
