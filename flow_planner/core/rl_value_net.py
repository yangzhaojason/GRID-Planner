from __future__ import annotations

import torch
import torch.nn as nn


class ValueMLP(nn.Module):
    """
    Simple baseline network V_phi(z) for trajectory-level returns.
    We keep it minimal and robust: input is a per-sample embedding z (B, D) and output is (B,).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = self.net(z).squeeze(-1)
        return v

