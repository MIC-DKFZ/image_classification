from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    entity: Optional[str] = None
    project: Optional[str] = None
    tags: Optional[List[str]] = None
    # W&B run group. Auto-generated from timestamp+model+peft if None.
    group: Optional[str] = None
    # W&B run name. Auto-generated if None.
    name: Optional[str] = None
    offline: bool = False
