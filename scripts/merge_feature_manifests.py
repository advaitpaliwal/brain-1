from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge processed feature manifests by concatenating features.")
    parser.add_argument("--left-manifest", required=True, help="Left processed manifest JSONL")
    parser.add_argument("--right-manifest", required=True, help="Right processed manifest JSONL")
    parser.add_argument("--output-manifest", required=True, help="Output merged manifest JSONL")
    parser.add_argument("--feature-root", required=True, help="Directory for merged feature tensors")
    args = parser.parse_args()

    left_manifest = Path(args.left_manifest).expanduser().resolve()
    right_manifest = Path(args.right_manifest).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    left_rows = [json.loads(line) for line in left_manifest.read_text(encoding="utf-8").splitlines()]
    right_rows = [json.loads(line) for line in right_manifest.read_text(encoding="utf-8").splitlines()]
    right_index = {
        (row["subject_id"], row["stimulus_id"]): row
        for row in right_rows
    }

    merged_rows: list[dict[str, object]] = []
    for left in left_rows:
        key = (left["subject_id"], left["stimulus_id"])
        if key not in right_index:
            continue
        right = right_index[key]

        left_features = torch.load(left["feature_path"], map_location="cpu")["features"].float()
        right_features = torch.load(right["feature_path"], map_location="cpu")["features"].float()
        left_target = torch.load(left["target_path"], map_location="cpu")["target"].float()
        right_target = torch.load(right["target_path"], map_location="cpu")["target"].float()

        length = min(left_features.shape[0], right_features.shape[0], left_target.shape[0], right_target.shape[0])
        merged_features = torch.cat([left_features[:length], right_features[:length]], dim=-1)
        target = left_target[:length]

        merged_feature_path = feature_root / f"{left['subject_id']}_{left['stimulus_id']}_features.pt"
        merged_target_path = feature_root / f"{left['subject_id']}_{left['stimulus_id']}_target.pt"
        torch.save({"features": merged_features}, merged_feature_path)
        torch.save({"target": target}, merged_target_path)

        merged_rows.append(
            {
                "dataset_name": f"{left['dataset_name']}+{right['dataset_name']}",
                "subject_id": left["subject_id"],
                "subject_index": left["subject_index"],
                "stimulus_id": left["stimulus_id"],
                "feature_path": str(merged_feature_path),
                "target_path": str(merged_target_path),
                "split": left.get("split", "train"),
            }
        )

    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"merged_rows={len(merged_rows)}")


if __name__ == "__main__":
    main()
