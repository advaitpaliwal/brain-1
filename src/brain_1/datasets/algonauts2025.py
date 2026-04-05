from __future__ import annotations

from brain_1.datasets.common import BaseDataset, SamplePaths


class Algonauts2025Dataset(BaseDataset):
    dataset_name = "algonauts2025"

    def build_manifest(self) -> list[SamplePaths]:
        return []

    def load_sample(self, sample: SamplePaths) -> dict:
        return {
            "dataset": self.dataset_name,
            "subject_id": sample.subject_id,
            "stimulus_id": sample.stimulus_id,
        }
