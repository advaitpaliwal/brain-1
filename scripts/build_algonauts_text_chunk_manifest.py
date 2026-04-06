from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a text-chunk manifest with inline TR texts.")
    parser.add_argument("--raw-manifest", required=True, help="Input raw text manifest JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--target-root",
        required=True,
        help="Root directory containing existing target tensors",
    )
    parser.add_argument(
        "--remote-target-root",
        default="",
        help="Optional rewritten root path for target tensors, e.g. /mnt/brain1-data/processed/...",
    )
    args = parser.parse_args()

    raw_manifest = Path(args.raw_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in raw_manifest.read_text(encoding="utf-8").splitlines()]
    chunk_rows: list[dict[str, object]] = []

    for row in rows:
        transcript_path = Path(row["transcript_path"])
        df = pd.read_csv(transcript_path, sep="\t")
        texts = df["text_per_tr"].fillna("").astype(str).tolist()

        target_name = f"{row['subject_id']}_{row['stimulus_id']}_target.pt"
        target_path = target_root / target_name
        if args.remote_target_root:
            target_path_str = str(Path(args.remote_target_root) / target_name)
        else:
            target_path_str = str(target_path)

        chunk_rows.append(
            {
                "dataset_name": row["dataset_name"],
                "subject_id": row["subject_id"],
                "subject_index": row["subject_index"],
                "split": row["split"],
                "season": row["season"],
                "stimulus_id": row["stimulus_id"],
                "texts": texts,
                "target_path": target_path_str,
            }
        )

    with output.open("w", encoding="utf-8") as handle:
        for row in chunk_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(chunk_rows)} chunk rows to {output}")


if __name__ == "__main__":
    main()
