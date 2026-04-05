from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class MultimodalManifestRecord:
    dataset_name: str
    subject_id: str
    subject_index: int
    stimulus_id: str
    text_feature_path: Path
    video_feature_path: Path
    target_path: Path


class MultimodalFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = self._load_records()

    def _load_records(self) -> list[MultimodalManifestRecord]:
        records: list[MultimodalManifestRecord] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records.append(
                    MultimodalManifestRecord(
                        dataset_name=row["dataset_name"],
                        subject_id=row["subject_id"],
                        subject_index=int(row["subject_index"]),
                        stimulus_id=row["stimulus_id"],
                        text_feature_path=Path(row["text_feature_path"]),
                        video_feature_path=Path(row["video_feature_path"]),
                        target_path=Path(row["target_path"]),
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        text = torch.load(record.text_feature_path, map_location="cpu")["features"].float()
        video = torch.load(record.video_feature_path, map_location="cpu")["features"].float()
        target = torch.load(record.target_path, map_location="cpu")["target"].float()
        length = min(text.shape[0], video.shape[0], target.shape[0])
        return {
            "dataset_name": record.dataset_name,
            "subject_id": record.subject_id,
            "subject_index": torch.tensor(record.subject_index, dtype=torch.long),
            "stimulus_id": record.stimulus_id,
            "text_features": text[:length],
            "video_features": video[:length],
            "target": target[:length],
            "length": length,
        }


def collate_multimodal_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(item["length"]) for item in batch]
    max_len = max(lengths)
    text_dim = batch[0]["text_features"].shape[-1]
    video_dim = batch[0]["video_features"].shape[-1]
    target_dim = batch[0]["target"].shape[-1]

    text = torch.zeros(len(batch), max_len, text_dim, dtype=torch.float32)
    video = torch.zeros(len(batch), max_len, video_dim, dtype=torch.float32)
    target = torch.zeros(len(batch), max_len, target_dim, dtype=torch.float32)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    target_mask = torch.zeros(len(batch), max_len, target_dim, dtype=torch.float32)

    for row_index, item in enumerate(batch):
        length = int(item["length"])
        text[row_index, :length] = item["text_features"]
        video[row_index, :length] = item["video_features"]
        target[row_index, :length] = item["target"]
        padding_mask[row_index, :length] = False
        target_mask[row_index, :length] = 1.0

    return {
        "text_features": text,
        "video_features": video,
        "target": target,
        "padding_mask": padding_mask,
        "target_mask": target_mask,
        "subject_index": torch.stack([item["subject_index"] for item in batch]),
    }
