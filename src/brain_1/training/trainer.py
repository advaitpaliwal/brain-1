from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from brain_1.datasets.common import (
    SyntheticTemporalDataset,
    TemporalFeatureDataset,
    collate_temporal_batch,
)
from brain_1.models import BrainModel, BrainModelConfig
from brain_1.training.losses import masked_mse
from brain_1.training.metrics import regression_metrics


@dataclass(slots=True)
class TrainingConfig:
    train_config_path: Path = Path("configs/train.yaml")
    model_config_path: Path = Path("configs/model.yaml")


def run_training(config: TrainingConfig) -> None:
    train_config = _load_yaml(config.train_config_path)
    model_config = _load_yaml(config.model_config_path)

    seed = int(train_config.get("seed", 1337))
    torch.manual_seed(seed)

    optimization = train_config["optimization"]
    data_config = train_config["data"]
    training_cfg = train_config["training"]
    backbone = model_config["backbone"]
    temporal_adapter = model_config["temporal_adapter"]
    head = model_config["head"]
    hrf = model_config["hrf"]
    device = torch.device(training_cfg.get("device", "cpu"))

    dataset = _build_dataset(
        manifest_path=Path(data_config["manifest_path"]),
        synthetic_fallback_size=int(data_config["synthetic_fallback_size"]),
        synthetic_seq_len=int(data_config["synthetic_seq_len"]),
        feature_dim=int(backbone["input_feature_dim"]),
        parcel_dim=int(head["output_dim"]),
    )
    raw_val_manifest = data_config.get("val_manifest_path")
    val_manifest_path = Path(raw_val_manifest) if raw_val_manifest else None
    val_dataset = (
        TemporalFeatureDataset(val_manifest_path)
        if val_manifest_path is not None and val_manifest_path.is_file()
        else None
    )
    inferred_feature_dim, inferred_parcel_dim = _infer_dimensions(
        dataset,
        default_feature_dim=int(backbone["input_feature_dim"]),
        default_parcel_dim=int(head["output_dim"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(optimization["batch_size"]),
        shuffle=True,
        collate_fn=collate_temporal_batch,
    )
    val_dataloader = (
        DataLoader(
            val_dataset,
            batch_size=int(optimization["batch_size"]),
            shuffle=False,
            collate_fn=collate_temporal_batch,
        )
        if val_dataset is not None
        else None
    )

    model = BrainModel(
        BrainModelConfig(
            input_feature_dim=inferred_feature_dim,
            hidden_size=int(model_config["projection"]["hidden_size"]),
            adapter_layers=int(temporal_adapter["layers"]),
            adapter_heads=int(temporal_adapter["heads"]),
            adapter_dropout=float(temporal_adapter["dropout"]),
            parcel_dim=inferred_parcel_dim,
            subject_count=max(32, _infer_subject_count(dataset)),
            subject_embedding_dim=int(head["subject_embedding_dim"]),
            hrf_kernel_size=int(hrf["kernel_size"]),
        )
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["lr"]),
        weight_decay=float(optimization["weight_decay"]),
    )

    max_steps = int(optimization["max_steps"])
    log_every = int(training_cfg["log_every"])
    eval_every = int(training_cfg.get("eval_every", 0))
    step = 0
    model.train()
    best_val_pearson = float("-inf")
    history: list[dict[str, float | int]] = []
    output_dir = Path(training_cfg.get("output_dir", "artifacts/default_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    while step < max_steps:
        for batch in dataloader:
            features = batch["features"].to(device)
            subject_index = batch["subject_index"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            pred = model(
                features=features,
                subject_index=subject_index,
                padding_mask=padding_mask,
            )
            loss = masked_mse(pred, target, mask=target_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip_norm = float(training_cfg["grad_clip_norm"])
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            if step % log_every == 0 or step == max_steps - 1:
                metrics = regression_metrics(pred.detach().cpu(), target.detach().cpu())
                print(
                    f"step={step:05d} loss={loss.item():.4f} "
                    f"mse={metrics['mse']:.4f} pearson={metrics['pearson']:.4f}"
                )
                history.append(
                    {
                        "step": step,
                        "train_loss": float(loss.item()),
                        "train_mse": float(metrics["mse"]),
                        "train_pearson": float(metrics["pearson"]),
                    }
                )

            if val_dataloader is not None and eval_every > 0 and (
                step % eval_every == 0 or step == max_steps - 1
            ):
                val_metrics = _evaluate_model(model, val_dataloader, device)
                print(
                    f"val step={step:05d} loss={val_metrics['loss']:.4f} "
                    f"mse={val_metrics['mse']:.4f} pearson={val_metrics['pearson']:.4f}"
                )
                history.append(
                    {
                        "step": step,
                        "val_loss": float(val_metrics["loss"]),
                        "val_mse": float(val_metrics["mse"]),
                        "val_pearson": float(val_metrics["pearson"]),
                    }
                )
                if val_metrics["pearson"] > best_val_pearson:
                    best_val_pearson = float(val_metrics["pearson"])
                    best_checkpoint_path = output_dir / "best.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "train_config": train_config,
                            "model_config": model_config,
                            "best_val_metrics": val_metrics,
                            "best_step": step,
                        },
                        best_checkpoint_path,
                    )
                    print(f"Saved improved best checkpoint to {best_checkpoint_path}")

            step += 1
            if step >= max_steps:
                break

    checkpoint_path = output_dir / "final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": train_config,
            "model_config": model_config,
        },
        checkpoint_path,
    )
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved history to {history_path}")
    print(f"Saved checkpoint to {checkpoint_path}")
    print("Training run finished.")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_dataset(
    manifest_path: Path,
    synthetic_fallback_size: int,
    synthetic_seq_len: int,
    feature_dim: int,
    parcel_dim: int,
):
    if manifest_path.exists():
        print(f"Using manifest dataset: {manifest_path}")
        return TemporalFeatureDataset(manifest_path)

    print(f"Manifest not found at {manifest_path}, using synthetic fallback dataset.")
    return SyntheticTemporalDataset(
        size=synthetic_fallback_size,
        seq_len=synthetic_seq_len,
        feature_dim=feature_dim,
        parcel_dim=parcel_dim,
    )


def _infer_subject_count(dataset) -> int:
    if hasattr(dataset, "records"):
        return len({record.subject_index for record in dataset.records})
    if hasattr(dataset, "size"):
        return 4
    return 32


def _infer_dimensions(dataset, default_feature_dim: int, default_parcel_dim: int) -> tuple[int, int]:
    try:
        sample = dataset[0]
    except Exception:
        return default_feature_dim, default_parcel_dim
    return sample["features"].shape[-1], sample["target"].shape[-1]


def _evaluate_model(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            pred = model(
                features=batch["features"].to(device),
                subject_index=batch["subject_index"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            )
            target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            loss = masked_mse(pred, target, mask=target_mask)
            total_loss += float(loss.item())
            total_batches += 1

            mask = target_mask.detach().cpu().bool()
            pred_cpu = pred.detach().cpu()
            target_cpu = target.detach().cpu()
            preds.append(pred_cpu[mask])
            targets.append(target_cpu[mask])

    pred_tensor = torch.cat(preds, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    metrics = regression_metrics(pred_tensor, target_tensor)
    metrics["loss"] = total_loss / max(total_batches, 1)
    model.train()
    return metrics
