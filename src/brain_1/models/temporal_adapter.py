from __future__ import annotations

import torch
from torch import nn


class TemporalAdapter(nn.Module):
    def __init__(self, hidden_size: int, layers: int = 4, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )

    def forward(
        self, sequence: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encoder(sequence, src_key_padding_mask=padding_mask)
