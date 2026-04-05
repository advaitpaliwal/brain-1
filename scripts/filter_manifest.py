from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a manifest by regex on a chosen field.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--output", required=True, help="Output manifest JSONL")
    parser.add_argument("--field", default="stimulus_id", help="Field to match")
    parser.add_argument("--regex", required=True, help="Regex to keep")
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.regex)

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    kept = [row for row in rows if pattern.search(str(row[args.field]))]

    with output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")

    print(f"kept_rows={len(kept)}")


if __name__ == "__main__":
    main()
