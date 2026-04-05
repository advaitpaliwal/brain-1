from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from brain_1.models.hrf_head import HRFConv1d
from brain_1.models.parcel_head import ParcelHead
from brain_1.models.temporal_adapter import TemporalAdapter


@dataclass(slots=True)
class MultimodalBrainModelConfig:
    text_dim: int
    video_dim: int
    hidden_size: int = 1024
    adapter_layers: int = 4
    adapter_heads: int = 8
    adapter_dropout: float = 0.1
    parcel_dim: int = 1000
    subject_count: int = 32
    subject_embedding_dim: int = 64
    hrf_kernel_size: int = 5


class MultimodalBrainModel(nn.Module):
    def __init__(self, config: MultimodalBrainModelConfig) -> None:
        super().__init__()
        self.text_projector = nn.Sequential(
            nn.LayerNorm(config.text_dim),
            nn.Linear(config.text_dim, config.hidden_size),
            nn.GELU(),
        )
        self.video_projector = nn.Sequential(
            nn.LayerNorm(config.video_dim),
            nn.Linear(config.video_dim, config.hidden_size),
            nn.GELU(),
        )
        self.text_gain = nn.Parameter(torch.tensor(1.0))
        self.video_gain = nn.Parameter(torch.tensor(1.0))
        self.subject_embedding = nn.Embedding(config.subject_count, config.subject_embedding_dim)
        self.subject_projection = nn.Linear(config.subject_embedding_dim, config.hidden_size)
        self.temporal_adapter = TemporalAdapter(
            hidden_size=config.hidden_size,
            layers=config.adapter_layers,
            heads=config.adapter_heads,
            dropout=config.adapter_dropout,
        )
        self.hrf = HRFConv1d(hidden_size=config.hidden_size, kernel_size=config.hrf_kernel_size)
        self.parcel_head = ParcelHead(hidden_size=config.hidden_size, output_dim=config.parcel_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        video_features: torch.Tensor,
        subject_index: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_hidden = self.text_projector(text_features) * self.text_gain
        video_hidden = self.video_projector(video_features) * self.video_gain
        sequence = text_hidden + video_hidden
        subject_bias = self.subject_projection(self.subject_embedding(subject_index)).unsqueeze(1)
        sequence = sequence + subject_bias
        sequence = self.temporal_adapter(sequence, padding_mask=padding_mask)
        sequence = self.hrf(sequence)
        return self.parcel_head(sequence)
