from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class SamplePaths:
    subject_id: str
    stimulus_id: str
    dataset_name: str
    target_path: Path
    text_path: Path | None = None
    audio_path: Path | None = None
    video_path: Path | None = None


class BaseDataset:
    dataset_name: str = "base"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def build_manifest(self) -> list[SamplePaths]:
        raise NotImplementedError

    def load_sample(self, sample: SamplePaths) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(slots=True)
class ManifestRecord:
    dataset_name: str
    subject_id: str
    subject_index: int
    stimulus_id: str
    feature_path: Path
    target_path: Path


class TemporalFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = self._load_records()

    def _load_records(self) -> list[ManifestRecord]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")

        records: list[ManifestRecord] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records.append(
                    ManifestRecord(
                        dataset_name=row["dataset_name"],
                        subject_id=row["subject_id"],
                        subject_index=int(row["subject_index"]),
                        stimulus_id=row["stimulus_id"],
                        feature_path=Path(row["feature_path"]),
                        target_path=Path(row["target_path"]),
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        feature_payload = torch.load(record.feature_path, map_location="cpu")
        target_payload = torch.load(record.target_path, map_location="cpu")

        features = feature_payload["features"].float()
        target = target_payload["target"].float()
        length = min(features.shape[0], target.shape[0])

        return {
            "dataset_name": record.dataset_name,
            "subject_id": record.subject_id,
            "subject_index": torch.tensor(record.subject_index, dtype=torch.long),
            "stimulus_id": record.stimulus_id,
            "features": features[:length],
            "target": target[:length],
            "length": length,
        }


class SyntheticTemporalDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        size: int = 16,
        seq_len: int = 32,
        feature_dim: int = 4096,
        parcel_dim: int = 1000,
        seed: int = 1337,
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.parcel_dim = parcel_dim
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(self.seed + index)
        features = torch.randn(self.seq_len, self.feature_dim, generator=generator)
        target = torch.randn(self.seq_len, self.parcel_dim, generator=generator)
        return {
            "dataset_name": "synthetic",
            "subject_id": f"sub-{index % 4:02d}",
            "subject_index": torch.tensor(index % 4, dtype=torch.long),
            "stimulus_id": f"stim-{index:04d}",
            "features": features,
            "target": target,
            "length": self.seq_len,
        }


def collate_temporal_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(item["length"]) for item in batch]
    max_len = max(lengths)
    feature_dim = batch[0]["features"].shape[-1]
    target_dim = max(int(item["target"].shape[-1]) for item in batch)

    features = torch.zeros(len(batch), max_len, feature_dim, dtype=torch.float32)
    target = torch.zeros(len(batch), max_len, target_dim, dtype=torch.float32)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    target_mask = torch.zeros(len(batch), max_len, target_dim, dtype=torch.float32)

    for row_index, item in enumerate(batch):
        length = int(item["length"])
        item_target_dim = int(item["target"].shape[-1])
        features[row_index, :length] = item["features"]
        target[row_index, :length, :item_target_dim] = item["target"]
        padding_mask[row_index, :length] = False
        target_mask[row_index, :length, :item_target_dim] = 1.0

    return {
        "features": features,
        "target": target,
        "padding_mask": padding_mask,
        "target_mask": target_mask,
        "subject_index": torch.stack([item["subject_index"] for item in batch]),
        "dataset_name": [item["dataset_name"] for item in batch],
        "metadata": [
            {
                "dataset_name": item["dataset_name"],
                "subject_id": item["subject_id"],
                "stimulus_id": item["stimulus_id"],
                "length": item["length"],
            }
            for item in batch
        ],
    }
