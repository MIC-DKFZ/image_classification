from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn


class BaseModel(nn.Module):
    """Minimal torch-only replacement for the legacy training base class."""

    def __init__(self, *args, **kwargs):
        super().__init__()

        config: dict = {}
        if args:
            first = args[0]
            if isinstance(first, Mapping):
                config.update(first)
            else:
                raise TypeError(
                    "BaseModel only accepts a mapping positional argument for legacy configs."
                )

        config.update(kwargs)
        self.hparams = dict(config)

        for key, value in config.items():
            setattr(self, key, value)
