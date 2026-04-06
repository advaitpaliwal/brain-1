from __future__ import annotations

import torch


def regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if mask is not None:
        keep = mask.bool()
        pred_flat = pred[keep]
        target_flat = target[keep]
    else:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

    mse = torch.mean((pred_flat - target_flat) ** 2).item()
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    denom = pred_centered.norm() * target_centered.norm()
    pearson = torch.tensor(0.0)
    if denom > 0:
        pearson = torch.dot(pred_centered, target_centered) / denom

    return {
        "mse": float(mse),
        "pearson": float(pearson.item()),
    }
