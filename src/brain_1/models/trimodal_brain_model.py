from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from brain_1.models.hrf_head import HRFConv1d
from brain_1.models.parcel_head import ParcelHead
from brain_1.models.temporal_adapter import TemporalAdapter


@dataclass(slots=True)
class TrimodalBrainModelConfig:
    text_dim: int
    audio_dim: int
    video_dim: int
    hidden_size: int = 1024
    adapter_layers: int = 4
    adapter_heads: int = 8
    adapter_dropout: float = 0.1
    parcel_dim: int = 1000
    subject_count: int = 32
    subject_embedding_dim: int = 64
    hrf_kernel_size: int = 5
    modality_dropout_p: float = 0.3
    unseen_subject_p: float = 0.1


class TrimodalBrainModel(nn.Module):
    def __init__(self, config: TrimodalBrainModelConfig) -> None:
        super().__init__()
        self.config = config
        self.text_projector = nn.Sequential(nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, config.hidden_size), nn.GELU())
        self.audio_projector = nn.Sequential(nn.LayerNorm(config.audio_dim), nn.Linear(config.audio_dim, config.hidden_size), nn.GELU())
        self.video_projector = nn.Sequential(nn.LayerNorm(config.video_dim), nn.Linear(config.video_dim, config.hidden_size), nn.GELU())
        self.text_gain = nn.Parameter(torch.tensor(1.0))
        self.audio_gain = nn.Parameter(torch.tensor(1.0))
        self.video_gain = nn.Parameter(torch.tensor(1.0))
        self.subject_embedding = nn.Embedding(config.subject_count, config.subject_embedding_dim)
        self.subject_projection = nn.Linear(config.subject_embedding_dim, config.hidden_size)
        self.unseen_subject_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.temporal_adapter = TemporalAdapter(hidden_size=config.hidden_size, layers=config.adapter_layers, heads=config.adapter_heads, dropout=config.adapter_dropout)
        self.hrf = HRFConv1d(hidden_size=config.hidden_size, kernel_size=config.hrf_kernel_size)
        self.parcel_head = ParcelHead(hidden_size=config.hidden_size, output_dim=config.parcel_dim)

    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor, video_features: torch.Tensor, subject_index: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        text_hidden = self.text_projector(text_features) * self.text_gain
        audio_hidden = self.audio_projector(audio_features) * self.audio_gain
        video_hidden = self.video_projector(video_features) * self.video_gain

        if self.training:
            text_mask = torch.rand(text_hidden.shape[0], device=text_hidden.device) < self.config.modality_dropout_p
            audio_mask = torch.rand(audio_hidden.shape[0], device=audio_hidden.device) < self.config.modality_dropout_p
            video_mask = torch.rand(video_hidden.shape[0], device=video_hidden.device) < self.config.modality_dropout_p
            all_masked = text_mask & audio_mask & video_mask
            if all_masked.any():
                video_mask = video_mask & ~all_masked
            text_hidden = text_hidden.masked_fill(text_mask[:, None, None], 0.0)
            audio_hidden = audio_hidden.masked_fill(audio_mask[:, None, None], 0.0)
            video_hidden = video_hidden.masked_fill(video_mask[:, None, None], 0.0)

        sequence = text_hidden + audio_hidden + video_hidden
        subject_bias = self.subject_projection(self.subject_embedding(subject_index)).unsqueeze(1)
        if self.training:
            unseen_mask = torch.rand(subject_bias.shape[0], device=subject_bias.device) < self.config.unseen_subject_p
            unseen_bias = self.unseen_subject_projection(sequence.mean(dim=1)).unsqueeze(1)
            subject_bias = torch.where(unseen_mask[:, None, None], unseen_bias, subject_bias)
        sequence = sequence + subject_bias
        sequence = self.temporal_adapter(sequence, padding_mask=padding_mask)
        sequence = self.hrf(sequence)
        return self.parcel_head(sequence)
