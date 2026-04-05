from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from brain_1.datasets.trimodal import TrimodalFeatureDataset, collate_trimodal_batch
from brain_1.models.trimodal_brain_model import TrimodalBrainModel, TrimodalBrainModelConfig
from brain_1.training.losses import combined_regression_loss, masked_mse
from brain_1.training.metrics import regression_metrics


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def evaluate(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            pred = model(
                text_features=batch["text_features"].to(device),
                audio_features=batch["audio_features"].to(device),
                video_features=batch["video_features"].to(device),
                subject_index=batch["subject_index"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            )
            target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            loss = masked_mse(pred, target, mask=target_mask)
            total_loss += float(loss.item())
            total_batches += 1
            mask = target_mask.detach().cpu().bool()
            preds.append(pred.detach().cpu()[mask])
            targets.append(target.detach().cpu()[mask])
    metrics = regression_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))
    metrics["loss"] = total_loss / max(total_batches, 1)
    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the trimodal brain model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    train_dataset = TrimodalFeatureDataset(config["data"]["manifest_path"])
    val_dataset = TrimodalFeatureDataset(config["data"]["val_manifest_path"])
    batch_size = int(config["optimization"]["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_trimodal_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_trimodal_batch)

    sample = train_dataset[0]
    subject_count = max(32, len({record.subject_index for record in train_dataset.records}))
    model = TrimodalBrainModel(
        TrimodalBrainModelConfig(
            text_dim=sample["text_features"].shape[-1],
            audio_dim=sample["audio_features"].shape[-1],
            video_dim=sample["video_features"].shape[-1],
            hidden_size=1024,
            adapter_layers=4,
            adapter_heads=8,
            parcel_dim=sample["target"].shape[-1],
            subject_count=subject_count,
            subject_embedding_dim=64,
            hrf_kernel_size=5,
        )
    )
    device = torch.device(config["training"].get("device", "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optimization"]["lr"]), weight_decay=float(config["optimization"]["weight_decay"]))
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_pearson = float("-inf")
    history: list[dict[str, float | int]] = []
    loss_cfg = config.get("loss", {})

    max_steps = int(config["optimization"]["max_steps"])
    log_every = int(config["training"]["log_every"])
    eval_every = int(config["training"]["eval_every"])
    step = 0
    model.train()

    while step < max_steps:
        for batch in train_loader:
            pred = model(
                text_features=batch["text_features"].to(device),
                audio_features=batch["audio_features"].to(device),
                video_features=batch["video_features"].to(device),
                subject_index=batch["subject_index"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            )
            target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            loss, loss_parts = combined_regression_loss(
                pred,
                target,
                mask=target_mask,
                mse_weight=float(loss_cfg.get("mse_weight", 1.0)),
                correlation_weight=float(loss_cfg.get("correlation_weight", 0.0)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
            optimizer.step()

            if step % log_every == 0 or step == max_steps - 1:
                metrics = regression_metrics(pred.detach().cpu(), target.detach().cpu())
                history.append({
                    "step": step,
                    "train_loss": float(loss.item()),
                    "train_mse_loss": float(loss_parts["mse_loss"]),
                    "train_corr_loss": float(loss_parts["corr_loss"]),
                    "train_mse": float(metrics["mse"]),
                    "train_pearson": float(metrics["pearson"]),
                })
                print(
                    f"step={step:05d} loss={loss.item():.4f} mse={metrics['mse']:.4f} "
                    f"pearson={metrics['pearson']:.4f} mse_loss={loss_parts['mse_loss']:.4f} "
                    f"corr_loss={loss_parts['corr_loss']:.4f}"
                )

            if step % eval_every == 0 or step == max_steps - 1:
                val_metrics = evaluate(model, val_loader, device)
                history.append({"step": step, **{f'val_{k}': float(v) for k, v in val_metrics.items()}})
                print(f"val step={step:05d} loss={val_metrics['loss']:.4f} mse={val_metrics['mse']:.4f} pearson={val_metrics['pearson']:.4f}")
                if val_metrics["pearson"] > best_val_pearson:
                    best_val_pearson = float(val_metrics["pearson"])
                    save_checkpoint({"model_state_dict": cpu_state_dict(model), "config": config}, output_dir / "best.pt")

            step += 1
            if step >= max_steps:
                break

    save_checkpoint({"model_state_dict": cpu_state_dict(model), "config": config}, output_dir / "final.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
