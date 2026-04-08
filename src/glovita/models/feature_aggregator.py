from __future__ import annotations

import torch


AGGREGATION_METHODS = ("cls_token", "avg", "sum", "mean_all", "joint")


def aggregated_feature_dim(embed_dim: int, method: str, features_are_tokens: bool) -> int:
    if not features_are_tokens:
        return embed_dim
    if method == "joint":
        return embed_dim * 2
    return embed_dim


def aggregate_features(x: torch.Tensor, method: str) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim != 3:
        raise ValueError(
            f"Feature aggregation expects a 2D feature tensor or 3D token tensor, got shape={tuple(x.shape)}"
        )

    if method == "cls_token":
        return x[:, 0]
    if method == "avg":
        return x[:, 1:].mean(dim=1)
    if method == "sum":
        return x[:, 1:].sum(dim=1)
    if method == "mean_all":
        return x.mean(dim=1)
    if method == "joint":
        return torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=1)

    raise ValueError(f"Unknown feature aggregation method: {method!r}")
