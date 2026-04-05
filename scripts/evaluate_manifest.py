from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from brain_1.datasets.common import TemporalFeatureDataset, collate_temporal_batch
from brain_1.models import BrainModel, BrainModelConfig
from brain_1.training.losses import masked_mse
from brain_1.training.metrics import regression_metrics


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def infer_dimensions(dataset: TemporalFeatureDataset) -> tuple[int, int]:
    sample = dataset[0]
    return sample["features"].shape[-1], sample["target"].shape[-1]


def infer_subject_count(dataset: TemporalFeatureDataset) -> int:
    return len({record.subject_index for record in dataset.records})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a brain-1 checkpoint on a processed manifest.")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint")
    parser.add_argument("--manifest", required=True, help="Processed manifest JSONL to evaluate")
    parser.add_argument("--model-config", required=True, help="Model config YAML")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or mps")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    model_config_path = Path(args.model_config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = TemporalFeatureDataset(manifest_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
    )

    model_cfg = load_yaml(model_config_path)
    feature_dim, parcel_dim = infer_dimensions(dataset)
    subject_count = max(32, infer_subject_count(dataset))
    model = BrainModel(
        BrainModelConfig(
            input_feature_dim=feature_dim,
            hidden_size=int(model_cfg["projection"]["hidden_size"]),
            adapter_layers=int(model_cfg["temporal_adapter"]["layers"]),
            adapter_heads=int(model_cfg["temporal_adapter"]["heads"]),
            adapter_dropout=float(model_cfg["temporal_adapter"]["dropout"]),
            parcel_dim=parcel_dim,
            subject_count=subject_count,
            subject_embedding_dim=int(model_cfg["head"]["subject_embedding_dim"]),
            hrf_kernel_size=int(model_cfg["hrf"]["kernel_size"]),
        )
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device)
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
    metrics["num_rows"] = len(dataset.records)

    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
