from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def flatten_mapping(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value


def mlflow_param_value(value: Any) -> str | float | int:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, str)):
        return value
    if value is None:
        return "None"
    return str(make_json_safe(value))


def sanitize_artifact_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "/"} else "_" for ch in name)
