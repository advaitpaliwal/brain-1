from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_seasons(value: str) -> list[int]:
    seasons: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(chunk))
    return sorted(set(seasons))


def run(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    user_bin = str(Path.home() / "Library/Python/3.13/bin")
    env["PATH"] = f"{user_bin}:{env['PATH']}"
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected Algonauts Friends video files.")
    parser.add_argument("--dataset-root", required=True, help="Algonauts DataLad dataset root")
    parser.add_argument(
        "--seasons",
        default="1",
        help="Season list or range, e.g. 1 or 1-2",
    )
    parser.add_argument(
        "--limit-per-season",
        type=int,
        default=0,
        help="Optional max number of files to fetch per season",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    movie_root = root / "stimuli" / "movies" / "friends"
    seasons = parse_seasons(args.seasons)
    targets: list[str] = []

    for season in seasons:
        season_dir = movie_root / f"s{season}"
        files = sorted(season_dir.glob("friends_*.mkv"))
        if args.limit_per_season > 0:
            files = files[: args.limit_per_season]
        targets.extend(str(file.relative_to(root)) for file in files)

    if not targets:
        raise SystemExit("No movie targets selected.")

    run(["datalad", "get", *targets], cwd=root)
    print(f"Fetched {len(targets)} movie files.")


if __name__ == "__main__":
    main()
