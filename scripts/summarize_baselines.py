from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark metric files.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("metrics", nargs="+", help="Metric files in the form name=path")
    args = parser.parse_args()

    summary: dict[str, dict] = {}
    for item in args.metrics:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        summary[name] = json.loads(path.read_text(encoding="utf-8"))

    best_name = max(summary, key=lambda key: summary[key].get("pearson", float("-inf")))
    payload = {
        "best_run": best_name,
        "runs": summary,
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
