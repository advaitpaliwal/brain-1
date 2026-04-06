from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import tempfile
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from brain_1.datasets.common import TemporalFeatureDataset, collate_temporal_batch
from brain_1.models import MultiDatasetBrainModel, MultiDatasetBrainModelConfig
from brain_1.training.losses import combined_regression_loss, masked_mse
from brain_1.training.metrics import regression_metrics


@dataclass(slots=True)
class TrainingConfig:
    train_config_path: Path = Path("configs/train.yaml")
    model_config_path: Path = Path("configs/model.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a mixed-dataset brain-1 model with dataset-specific heads.")
    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
        help="Path to model config YAML",
    )
    args = parser.parse_args()
    run_training(
        TrainingConfig(
            train_config_path=Path(args.train_config),
            model_config_path=Path(args.model_config),
        )
    )


def run_training(config: TrainingConfig) -> None:
    train_config = _load_yaml(config.train_config_path)
    model_config = _load_yaml(config.model_config_path)

    seed = int(train_config.get("seed", 1337))
    torch.manual_seed(seed)

    optimization = train_config["optimization"]
    data_config = train_config["data"]
    loss_config = train_config.get("loss", {})
    training_cfg = train_config["training"]
    backbone = model_config["backbone"]
    temporal_adapter = model_config["temporal_adapter"]
    head = model_config["head"]
    hrf = model_config["hrf"]
    device = torch.device(training_cfg.get("device", "cpu"))

    manifest_path = Path(data_config["manifest_path"])
    dataset = TemporalFeatureDataset(manifest_path)
    raw_val_manifest = data_config.get("val_manifest_path")
    val_manifest_path = Path(raw_val_manifest) if raw_val_manifest else None
    val_dataset = (
        TemporalFeatureDataset(val_manifest_path)
        if val_manifest_path is not None and val_manifest_path.is_file()
        else None
    )

    inferred_feature_dim = dataset[0]["features"].shape[-1]
    dataset_output_dims = _infer_dataset_output_dims(dataset, val_dataset)
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

    model = MultiDatasetBrainModel(
        MultiDatasetBrainModelConfig(
            input_feature_dim=inferred_feature_dim,
            hidden_size=int(model_config["projection"]["hidden_size"]),
            adapter_layers=int(temporal_adapter["layers"]),
            adapter_heads=int(temporal_adapter["heads"]),
            adapter_dropout=float(temporal_adapter["dropout"]),
            dataset_output_dims=dataset_output_dims,
            subject_count=max(32, _infer_subject_count(dataset, val_dataset)),
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
    output_dir = Path(training_cfg.get("output_dir", "artifacts/default_mixed_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    while step < max_steps:
        for batch in dataloader:
            pred = model(
                features=batch["features"].to(device),
                subject_index=batch["subject_index"].to(device),
                dataset_names=batch["dataset_name"],
                padding_mask=batch["padding_mask"].to(device),
            )
            target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            loss, loss_parts = combined_regression_loss(
                pred,
                target,
                mask=target_mask,
                mse_weight=float(loss_config.get("mse_weight", 1.0)),
                correlation_weight=float(loss_config.get("correlation_weight", 0.0)),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip_norm"]))
            optimizer.step()

            if step % log_every == 0 or step == max_steps - 1:
                metrics = regression_metrics(
                    pred.detach().cpu(),
                    target.detach().cpu(),
                    mask=target_mask.detach().cpu().bool(),
                )
                print(
                    f"step={step:05d} loss={loss.item():.4f} "
                    f"mse={metrics['mse']:.4f} pearson={metrics['pearson']:.4f} "
                    f"mse_loss={loss_parts['mse_loss']:.4f} corr_loss={loss_parts['corr_loss']:.4f}"
                )
                history.append(
                    {
                        "step": step,
                        "train_loss": float(loss.item()),
                        "train_mse_loss": float(loss_parts["mse_loss"]),
                        "train_corr_loss": float(loss_parts["corr_loss"]),
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
                    _save_checkpoint(
                        {
                            "model_state_dict": _cpu_state_dict(model),
                            "train_config": train_config,
                            "model_config": model_config,
                            "dataset_output_dims": dataset_output_dims,
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
    _save_checkpoint(
        {
            "model_state_dict": _cpu_state_dict(model),
            "train_config": train_config,
            "model_config": model_config,
            "dataset_output_dims": dataset_output_dims,
        },
        checkpoint_path,
    )
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved history to {history_path}")
    print(f"Saved checkpoint to {checkpoint_path}")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _infer_dataset_output_dims(*datasets: TemporalFeatureDataset | None) -> dict[str, int]:
    output_dims: dict[str, int] = {}
    for dataset in datasets:
        if dataset is None:
            continue
        for record in dataset.records:
            if record.dataset_name in output_dims:
                continue
            target = torch.load(record.target_path, map_location="cpu")["target"]
            output_dims[record.dataset_name] = int(target.shape[-1])
    return output_dims


def _infer_subject_count(*datasets: TemporalFeatureDataset | None) -> int:
    subject_indices: list[int] = []
    for dataset in datasets:
        if dataset is None:
            continue
        subject_indices.extend(int(record.subject_index) for record in dataset.records)
    return (max(subject_indices) + 1) if subject_indices else 32


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
                dataset_names=batch["dataset_name"],
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


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
