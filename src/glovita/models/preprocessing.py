from __future__ import annotations

from dataclasses import dataclass
import re

from glovita.configs.model import (
    Dinov2EncoderConfig,
    Dinov3EncoderConfig,
    EncoderConfig,
    PrecomputedEncoderConfig,
    PrimusEncoderConfig,
    TimmEncoderConfig,
    TorchvisionEncoderConfig,
    TransformerEncoderConfig,
)


@dataclass(frozen=True)
class PreprocessingDefaults:
    image_size: int | None = None
    resize_size: int | None = None
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    patch_size: tuple[int, int, int] | None = None

    def as_kwargs(self) -> dict:
        return {
            "image_size": self.image_size,
            "resize_size": self.resize_size,
            "mean": self.mean,
            "std": self.std,
            "patch_size": self.patch_size,
        }


def _safe_tuple(value) -> tuple | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _safe_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _safe_int(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_patch_size_from_name(name: str) -> tuple[int, int] | None:
    match = re.search(r"(?:patch|vit[blsgh]?)(\d+)", name)
    if match is None:
        match = re.search(r"p(\d+)", name)
    if match is None:
        return None
    patch = int(match.group(1))
    return (patch, patch)


def _infer_timm_patch_size(model) -> tuple[int, int] | None:
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is None:
        return None
    patch_size = getattr(patch_embed, "patch_size", None)
    if patch_size is None and hasattr(patch_embed, "proj"):
        patch_size = getattr(patch_embed.proj, "kernel_size", None)
    patch_size = _safe_tuple(patch_size)
    if patch_size is None:
        return None
    if len(patch_size) == 1:
        return (int(patch_size[0]), int(patch_size[0]))
    return tuple(int(v) for v in patch_size[:2])


def _infer_transformers_patch_size(model) -> tuple[int, int] | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    for attr in ("patch_size", "encoder_stride"):
        value = getattr(config, attr, None)
        if value is not None:
            patch_size = _safe_tuple(value)
            if patch_size is None:
                continue
            if len(patch_size) == 1:
                return (int(patch_size[0]), int(patch_size[0]))
            return tuple(int(v) for v in patch_size[:2])
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        patch_size = getattr(vision_config, "patch_size", None)
        patch_size = _safe_tuple(patch_size)
        if patch_size is not None:
            if len(patch_size) == 1:
                return (int(patch_size[0]), int(patch_size[0]))
            return tuple(int(v) for v in patch_size[:2])
    return None


def _infer_transformers_patch_size_from_processor(processor) -> tuple[int, int] | None:
    for attr in ("patch_size",):
        value = getattr(processor, attr, None)
        if value is not None:
            patch_size = _safe_tuple(value)
            if patch_size is None:
                continue
            if len(patch_size) == 1:
                return (int(patch_size[0]), int(patch_size[0]))
            return tuple(int(v) for v in patch_size[:2])
    return None


def resolve_encoder_preprocessing_defaults(encoder_config: EncoderConfig) -> PreprocessingDefaults:
    if isinstance(encoder_config, TimmEncoderConfig):
        try:
            import timm
            from timm.data import resolve_model_data_config

            model = timm.create_model(encoder_config.type, pretrained=False, num_classes=0)
            patch_size = _infer_timm_patch_size(model)
            if not encoder_config.pretrained:
                return PreprocessingDefaults(patch_size=patch_size)
            data_cfg = resolve_model_data_config(model=model)
            input_size = _safe_tuple(data_cfg.get("input_size"))
            image_size = int(input_size[-1]) if input_size is not None else None
            crop_pct = float(data_cfg.get("crop_pct", 1.0))
            resize_size = int(round(image_size / crop_pct)) if image_size is not None and crop_pct > 0 else None
            return PreprocessingDefaults(
                image_size=image_size,
                resize_size=resize_size,
                mean=_safe_tuple(data_cfg.get("mean")),
                std=_safe_tuple(data_cfg.get("std")),
                patch_size=patch_size,
            )
        except Exception:
            return PreprocessingDefaults()

    if isinstance(encoder_config, TorchvisionEncoderConfig) and encoder_config.pretrained:
        try:
            import torchvision.models as tvm

            weights = tvm.get_model_weights(encoder_config.type).DEFAULT
            transforms = weights.transforms()
            crop_size = getattr(transforms, "crop_size", None)
            resize_size = getattr(transforms, "resize_size", None)
            mean = getattr(transforms, "mean", None)
            std = getattr(transforms, "std", None)
            crop_size = crop_size[0] if isinstance(crop_size, (list, tuple)) else crop_size
            resize_size = resize_size[0] if isinstance(resize_size, (list, tuple)) else resize_size
            model = tvm.get_model(encoder_config.type, weights=None)
            patch_size = None
            if hasattr(model, "patch_size"):
                patch_size = _safe_tuple(getattr(model, "patch_size"))
            elif hasattr(model, "conv_proj"):
                patch_size = _safe_tuple(getattr(model.conv_proj, "kernel_size", None))
            return PreprocessingDefaults(
                image_size=int(crop_size) if crop_size is not None else None,
                resize_size=int(resize_size) if resize_size is not None else None,
                mean=_safe_tuple(mean),
                std=_safe_tuple(std),
                patch_size=patch_size,
            )
        except Exception:
            return PreprocessingDefaults()

    if isinstance(encoder_config, TransformerEncoderConfig) and encoder_config.pretrained:
        try:
            from transformers import AutoImageProcessor

            processor = AutoImageProcessor.from_pretrained(encoder_config.type)
            size = getattr(processor, "size", None)
            image_size = None
            resize_size = None
            if isinstance(size, dict):
                image_size = size.get("height") or size.get("width") or size.get("shortest_edge")
                resize_size = size.get("shortest_edge") or image_size
            elif isinstance(size, int):
                image_size = size
                resize_size = size
            patch_size = _infer_transformers_patch_size_from_processor(processor)
            if patch_size is None:
                model = AutoModel.from_pretrained(encoder_config.type) if encoder_config.pretrained else None
                patch_size = _infer_transformers_patch_size(model) if model is not None else None
            return PreprocessingDefaults(
                image_size=int(image_size) if image_size is not None else None,
                resize_size=int(resize_size) if resize_size is not None else None,
                mean=_safe_tuple(getattr(processor, "image_mean", None)),
                std=_safe_tuple(getattr(processor, "image_std", None)),
                patch_size=patch_size,
            )
        except Exception:
            return PreprocessingDefaults()

    if isinstance(encoder_config, Dinov2EncoderConfig):
        return PreprocessingDefaults(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            patch_size=_extract_patch_size_from_name(encoder_config.type),
        )

    if isinstance(encoder_config, Dinov3EncoderConfig):
        return PreprocessingDefaults(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            patch_size=_extract_patch_size_from_name(encoder_config.type),
        )

    if isinstance(encoder_config, PrimusEncoderConfig):
        return PreprocessingDefaults(
            patch_size=tuple(int(v) for v in encoder_config.input_shape),
        )

    if isinstance(encoder_config, PrecomputedEncoderConfig):
        return PreprocessingDefaults()

    return PreprocessingDefaults()
