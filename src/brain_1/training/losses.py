from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom
