from __future__ import annotations

import json
import sys
from typing import Any, Sequence, TypeVar

import tyro


T = TypeVar("T")


def _parse_add_log_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _insert_nested_log_value(root: dict[str, Any], path: list[str], value: Any) -> None:
    current = root
    for index, part in enumerate(path):
        is_leaf = index == len(path) - 1
        if is_leaf:
            existing = current.get(part)
            if isinstance(existing, dict):
                existing["value"] = value
            else:
                current[part] = value
            return

        existing = current.get(part)
        if existing is None:
            child: dict[str, Any] = {}
            current[part] = child
            current = child
            continue
        if isinstance(existing, dict):
            current = existing
            continue

        child = {"value": existing}
        current[part] = child
        current = child


def _extract_add_log_args(args: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    remaining: list[str] = []
    add_log: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--add_log.") or token.startswith("--add-log."):
            if "=" in token:
                flag, raw_value = token.split("=", 1)
            else:
                if i + 1 >= len(args):
                    raise ValueError(f"Missing value for logging-only CLI flag {token!r}.")
                flag = token
                raw_value = args[i + 1]
                i += 1

            if flag.startswith("--add_log."):
                dotted_path = flag[len("--add_log.") :]
            else:
                dotted_path = flag[len("--add-log.") :]
            path = [segment for segment in dotted_path.split(".") if segment]
            if not path:
                raise ValueError(f"Invalid add_log path in CLI flag {flag!r}.")
            _insert_nested_log_value(add_log, path, _parse_add_log_value(raw_value))
        else:
            remaining.append(token)
        i += 1
    return remaining, add_log


def parse_cli(config_type: type[T], args: Sequence[str] | None = None) -> T:
    cli_args = list(sys.argv[1:] if args is None else args)
    return tyro.cli(config_type, args=cli_args, use_underscores=True)


def parse_root_cli(config_type: type[T], args: Sequence[str] | None = None) -> T:
    cli_args = list(sys.argv[1:] if args is None else args)
    filtered_args, add_log = _extract_add_log_args(cli_args)
    config = tyro.cli(config_type, args=filtered_args, use_underscores=True)
    if add_log:
        config = config.model_copy(update={"add_log": add_log})
    return config
