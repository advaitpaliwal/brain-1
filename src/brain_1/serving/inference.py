from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InferenceResult:
    shape: tuple[int, int]
    description: str


def run_inference() -> InferenceResult:
    return InferenceResult(
        shape=(0, 1000),
        description="Inference pipeline not implemented yet.",
    )
