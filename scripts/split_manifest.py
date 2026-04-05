from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a processed manifest into train/val sets.")
    parser.add_argument("--manifest", required=True, help="Input processed manifest JSONL")
    parser.add_argument("--train-output", required=True, help="Output train manifest JSONL")
    parser.add_argument("--val-output", required=True, help="Output val manifest JSONL")
    parser.add_argument(
        "--val-regex",
        default="",
        help="Regex applied to stimulus_id to select validation rows",
    )
    parser.add_argument(
        "--val-stimuli",
        default="",
        help="Comma-separated list of stimulus ids to hold out for validation",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    train_output = Path(args.train_output).expanduser().resolve()
    val_output = Path(args.val_output).expanduser().resolve()
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(args.val_regex) if args.val_regex else None
    explicit_stimuli = {value.strip() for value in args.val_stimuli.split(",") if value.strip()}

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for row in rows:
        stimulus_id = row["stimulus_id"]
        is_val = stimulus_id in explicit_stimuli or (pattern is not None and pattern.search(stimulus_id))
        (val_rows if is_val else train_rows).append(row)

    with train_output.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row) + "\n")

    with val_output.open("w", encoding="utf-8") as handle:
        for row in val_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}")


if __name__ == "__main__":
    main()
