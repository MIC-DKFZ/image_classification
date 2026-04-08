from __future__ import annotations

import torch.nn as nn
from timm.layers import ClassifierHead as TimmClassifierHead


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.fc = TimmClassifierHead(input_dim, num_classes, pool_type="", drop_rate=dropout)

    def forward(self, x):
        return self.fc(x)
