from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Placeholder downloader for OpenNeuro-backed brain-1 datasets."
    )
    parser.add_argument("--dataset", required=True, help="OpenNeuro dataset id, e.g. ds003020")
    parser.add_argument("--output", required=True, help="Destination directory")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"Prepared output directory: {output}")
    print(f"Next step: implement OpenNeuro download flow for dataset {args.dataset}.")


if __name__ == "__main__":
    main()
