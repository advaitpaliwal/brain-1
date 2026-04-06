from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate manifest JSONL files and remap subject indices.")
    parser.add_argument(
        "--manifests",
        required=True,
        nargs="+",
        help="Input manifest JSONL files",
    )
    parser.add_argument("--output", required=True, help="Output JSONL manifest")
    parser.add_argument(
        "--starting-index",
        type=int,
        default=0,
        help="First subject index to assign when remapping",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for manifest in args.manifests:
        path = Path(manifest).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))

    subject_map: dict[tuple[str, str], int] = {}
    next_index = args.starting_index
    for row in rows:
        key = (str(row["dataset_name"]), str(row["subject_id"]))
        if key not in subject_map:
            subject_map[key] = next_index
            next_index += 1
        row["subject_index"] = subject_map[key]

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(
        f"Wrote {len(rows)} rows to {output} "
        f"with {len(subject_map)} unique subject embeddings"
    )


if __name__ == "__main__":
    main()
