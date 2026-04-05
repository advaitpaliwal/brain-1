from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from brain_1.models.backbone_qwen import QwenBackboneConfig, QwenFeatureExtractor
from brain_1.models.hrf_head import HRFConv1d
from brain_1.models.parcel_head import ParcelHead
from brain_1.models.temporal_adapter import TemporalAdapter


@dataclass(slots=True)
class BrainModelConfig:
    input_feature_dim: int = 4096
    hidden_size: int = 1024
    adapter_layers: int = 4
    adapter_heads: int = 8
    adapter_dropout: float = 0.1
    parcel_dim: int = 1000
    subject_count: int = 32
    subject_embedding_dim: int = 64
    hrf_kernel_size: int = 5


class BrainModel(nn.Module):
    def __init__(self, config: BrainModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = QwenFeatureExtractor(
            QwenBackboneConfig(hidden_size=config.input_feature_dim)
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(config.input_feature_dim),
            nn.Linear(config.input_feature_dim, config.hidden_size),
            nn.GELU(),
        )
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
        features: torch.Tensor,
        subject_index: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        backbone_features = self.backbone({"features": features, "padding_mask": padding_mask})
        sequence = self.projector(backbone_features.sequence)

        subject_embed = self.subject_embedding(subject_index)
        subject_bias = self.subject_projection(subject_embed).unsqueeze(1)
        sequence = sequence + subject_bias

        sequence = self.temporal_adapter(sequence, padding_mask=backbone_features.padding_mask)
        sequence = self.hrf(sequence)
        return self.parcel_head(sequence)
