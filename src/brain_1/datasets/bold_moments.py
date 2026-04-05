from __future__ import annotations

from brain_1.datasets.common import BaseDataset, SamplePaths


class BoldMomentsDataset(BaseDataset):
    dataset_name = "bold_moments"

    def build_manifest(self) -> list[SamplePaths]:
        return []

    def load_sample(self, sample: SamplePaths) -> dict:
        return {
            "dataset": self.dataset_name,
            "subject_id": sample.subject_id,
            "stimulus_id": sample.stimulus_id,
        }
