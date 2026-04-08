from __future__ import annotations

import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
