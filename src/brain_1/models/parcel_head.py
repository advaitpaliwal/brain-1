from __future__ import annotations

import torch
from torch import nn


class ParcelHead(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int = 1000) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.proj(sequence)
