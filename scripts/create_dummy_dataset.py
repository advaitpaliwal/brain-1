from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset for brain-1.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path")
    parser.add_argument("--count", type=int, default=16, help="Number of samples to create")
    parser.add_argument("--seq-len", type=int, default=32, help="Temporal length")
    parser.add_argument("--feature-dim", type=int, default=4096, help="Feature dimension")
    parser.add_argument("--parcel-dim", type=int, default=1000, help="Target parcel dimension")
    args = parser.parse_args()

    manifest_path = Path(args.output).expanduser().resolve()
    root = manifest_path.parent
    feature_dir = root / "features"
    target_dir = root / "targets"
    feature_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    generator = torch.Generator().manual_seed(1337)

    for index in range(args.count):
        features = torch.randn(args.seq_len, args.feature_dim, generator=generator)
        target = torch.randn(args.seq_len, args.parcel_dim, generator=generator)

        feature_path = feature_dir / f"sample_{index:04d}.pt"
        target_path = target_dir / f"sample_{index:04d}.pt"
        torch.save({"features": features}, feature_path)
        torch.save({"target": target}, target_path)

        rows.append(
            {
                "dataset_name": "synthetic",
                "subject_id": f"sub-{index % 4:02d}",
                "subject_index": index % 4,
                "stimulus_id": f"stim-{index:04d}",
                "feature_path": str(feature_path),
                "target_path": str(target_path),
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} synthetic samples to {manifest_path}")


if __name__ == "__main__":
    main()
