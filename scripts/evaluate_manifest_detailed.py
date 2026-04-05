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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pearson_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom <= 0:
        return float("nan")
    return float(torch.dot(x, y) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed parcel/subject evaluation for brain-1.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    model_config_path = Path(args.model_config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = TemporalFeatureDataset(manifest_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_temporal_batch)
    model_cfg = load_yaml(model_config_path)
    sample = dataset[0]
    subject_count = max(32, len({record.subject_index for record in dataset.records}))

    model = BrainModel(
        BrainModelConfig(
            input_feature_dim=sample["features"].shape[-1],
            hidden_size=int(model_cfg["projection"]["hidden_size"]),
            adapter_layers=int(model_cfg["temporal_adapter"]["layers"]),
            adapter_heads=int(model_cfg["temporal_adapter"]["heads"]),
            adapter_dropout=float(model_cfg["temporal_adapter"]["dropout"]),
            parcel_dim=sample["target"].shape[-1],
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

    by_subject: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    total_loss = 0.0
    total_batches = 0

    row_index = 0
    with torch.no_grad():
        for batch in dataloader:
            pred = model(
                features=batch["features"].to(device),
                subject_index=batch["subject_index"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            ).cpu()
            target = batch["target"]
            target_mask = batch["target_mask"]
            loss = masked_mse(pred, target, mask=target_mask)
            total_loss += float(loss.item())
            total_batches += 1

            for sample_idx, meta in enumerate(batch["metadata"]):
                length = int(meta["length"])
                subject_id = str(meta["subject_id"])
                pred_slice = pred[sample_idx, :length]
                target_slice = target[sample_idx, :length]
                by_subject.setdefault(subject_id, []).append((pred_slice, target_slice))
                row_index += 1

    subject_scores: dict[str, float] = {}
    parcel_scores: list[float] = []
    parcel_dim = sample["target"].shape[-1]

    for subject_id, pairs in by_subject.items():
        pred_cat = torch.cat([pred for pred, _ in pairs], dim=0)
        target_cat = torch.cat([target for _, target in pairs], dim=0)
        subject_scores[subject_id] = pearson_1d(pred_cat.flatten(), target_cat.flatten())

    all_pred = torch.cat([torch.cat([pred for pred, _ in pairs], dim=0) for pairs in by_subject.values()], dim=0)
    all_target = torch.cat([torch.cat([target for _, target in pairs], dim=0) for pairs in by_subject.values()], dim=0)

    for parcel_idx in range(parcel_dim):
        parcel_scores.append(pearson_1d(all_pred[:, parcel_idx], all_target[:, parcel_idx]))

    valid_parcel_scores = [score for score in parcel_scores if score == score]
    mean_parcel_pearson = sum(valid_parcel_scores) / max(len(valid_parcel_scores), 1)
    mean_subject_pearson = sum(subject_scores.values()) / max(len(subject_scores), 1)

    payload = {
        "loss": total_loss / max(total_batches, 1),
        "mean_parcel_pearson": mean_parcel_pearson,
        "mean_subject_pearson": mean_subject_pearson,
        "subject_scores": subject_scores,
        "num_parcels": parcel_dim,
        "num_rows": len(dataset.records),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
