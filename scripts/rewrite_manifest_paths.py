from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite absolute manifest paths under a new root.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--source-root", required=True, help="Current local root prefix")
    parser.add_argument("--dest-root", required=True, help="Destination root prefix")
    parser.add_argument("--output", required=True, help="Output manifest JSONL")
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rewritten: list[dict] = []

    for row in rows:
        new_row = dict(row)
        for key, value in list(row.items()):
            if not key.endswith("_path"):
                continue
            path = Path(value)
            if path.is_absolute():
                try:
                    rel = path.resolve().relative_to(source_root)
                except Exception:
                    continue
                new_row[key] = str(dest_root / rel)
        rewritten.append(new_row)

    with output.open("w", encoding="utf-8") as handle:
        for row in rewritten:
            handle.write(json.dumps(row) + "\n")

    print(f"rewrote_rows={len(rewritten)}")


if __name__ == "__main__":
    main()
