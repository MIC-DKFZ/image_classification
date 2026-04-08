from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TransformBuilder = Callable[..., object]


@dataclass(frozen=True)
class TrainPolicySpec:
    build: TransformBuilder
    default_kwargs: dict = field(default_factory=dict)
