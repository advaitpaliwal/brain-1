from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def parse_csv(value: str) -> list[str]:
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone a git-annex dataset and materialize selected files.")
    parser.add_argument("--repo-url", required=True, help="Git URL of the annex-backed repository")
    parser.add_argument("--output", required=True, help="Destination checkout directory")
    parser.add_argument("--paths", default="", help="Comma-separated relative paths to git-annex get")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    run(["git", "clone", args.repo_url, str(output)])
    run(["git", "config", "user.email", "brain1@local.invalid"], cwd=output)
    run(["git", "config", "user.name", "brain-1"], cwd=output)
    run(["git", "annex", "init", "brain-1"], cwd=output)

    paths = parse_csv(args.paths)
    if paths:
        run(["git", "annex", "get", *paths], cwd=output)

    print(f"Prepared repo at {output}")
    if paths:
        print(f"Materialized {len(paths)} path(s)")


if __name__ == "__main__":
    main()
