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


def masked_pearson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if mask is None:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
    else:
        keep = mask.bool()
        pred_flat = pred[keep]
        target_flat = target[keep]

    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    denom = pred_centered.norm() * target_centered.norm()
    pearson = torch.dot(pred_centered, target_centered) / denom.clamp_min(eps)
    return 1.0 - pearson


def combined_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    mse_weight: float = 1.0,
    correlation_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = masked_mse(pred, target, mask=mask)
    correlation = masked_pearson_loss(pred, target, mask=mask)
    total = mse_weight * mse + correlation_weight * correlation
    return total, {
        "mse_loss": float(mse.detach().cpu().item()),
        "corr_loss": float(correlation.detach().cpu().item()),
    }
