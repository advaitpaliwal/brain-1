from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class QwenBackboneConfig:
    pretrained_id: str = "Qwen/Qwen2.5-Omni-7B"
    hidden_size: int = 4096
    freeze: bool = True


@dataclass(slots=True)
class BackboneFeatures:
    sequence: torch.Tensor
    padding_mask: torch.Tensor | None = None


class QwenFeatureExtractor(nn.Module):
    """Stub multimodal feature extractor.

    The real implementation should:
    - load a commercial-friendly multimodal backbone
    - expose hidden states on a shared temporal grid
    - keep modality-specific preprocessing outside the model core
    """

    def __init__(self, config: QwenBackboneConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: dict[str, torch.Tensor]) -> BackboneFeatures:
        if "features" not in batch:
            raise KeyError("Expected 'features' tensor in batch.")
        sequence = batch["features"]
        return BackboneFeatures(sequence=sequence, padding_mask=batch.get("padding_mask"))
