from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import nn

from brain_1.models.backbone_qwen import QwenBackboneConfig, QwenFeatureExtractor
from brain_1.models.hrf_head import HRFConv1d
from brain_1.models.parcel_head import ParcelHead
from brain_1.models.temporal_adapter import TemporalAdapter


def _sanitize_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name)


@dataclass(slots=True)
class MultiDatasetBrainModelConfig:
    input_feature_dim: int = 4096
    hidden_size: int = 1024
    adapter_layers: int = 4
    adapter_heads: int = 8
    adapter_dropout: float = 0.1
    dataset_output_dims: dict[str, int] | None = None
    subject_count: int = 32
    subject_embedding_dim: int = 64
    hrf_kernel_size: int = 5


class MultiDatasetBrainModel(nn.Module):
    def __init__(self, config: MultiDatasetBrainModelConfig) -> None:
        super().__init__()
        if not config.dataset_output_dims:
            raise ValueError("dataset_output_dims must be provided for MultiDatasetBrainModel")

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

        self.dataset_key_map = {
            dataset_name: _sanitize_key(dataset_name)
            for dataset_name in sorted(config.dataset_output_dims)
        }
        self.dataset_output_dims = dict(config.dataset_output_dims)
        self.dataset_heads = nn.ModuleDict(
            {
                self.dataset_key_map[dataset_name]: ParcelHead(
                    hidden_size=config.hidden_size,
                    output_dim=output_dim,
                )
                for dataset_name, output_dim in sorted(config.dataset_output_dims.items())
            }
        )

    def forward(
        self,
        features: torch.Tensor,
        subject_index: torch.Tensor,
        dataset_names: list[str],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.shape[0] != len(dataset_names):
            raise ValueError("dataset_names length must match batch size")

        backbone_features = self.backbone({"features": features, "padding_mask": padding_mask})
        sequence = self.projector(backbone_features.sequence)

        subject_embed = self.subject_embedding(subject_index)
        subject_bias = self.subject_projection(subject_embed).unsqueeze(1)
        sequence = sequence + subject_bias

        sequence = self.temporal_adapter(sequence, padding_mask=backbone_features.padding_mask)
        sequence = self.hrf(sequence)

        max_output_dim = max(self.dataset_output_dims[name] for name in dataset_names)
        pred = sequence.new_zeros(sequence.shape[0], sequence.shape[1], max_output_dim)
        for row_index, dataset_name in enumerate(dataset_names):
            head = self.dataset_heads[self.dataset_key_map[dataset_name]]
            row_pred = head(sequence[row_index : row_index + 1])[0]
            pred[row_index, :, : row_pred.shape[-1]] = row_pred
        return pred
