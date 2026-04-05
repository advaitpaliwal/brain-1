from __future__ import annotations

import torch
from torch import nn


class HRFConv1d(nn.Module):
    """Small temporal smoothing layer to emulate hemodynamic blur."""

    def __init__(self, hidden_size: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_size,
            bias=False,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = sequence.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)
