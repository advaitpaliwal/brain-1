from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/courtois-neuromod/algonauts_2025.competitors.git"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    env = os.environ.copy()
    user_bin = str(Path.home() / "Library/Python/3.13/bin")
    env["PATH"] = f"{user_bin}:{env['PATH']}"
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a minimal Algonauts 2025 slice with DataLad.")
    parser.add_argument(
        "--output",
        default="data/raw/algonauts2025",
        help="Destination directory for the DataLad dataset",
    )
    parser.add_argument("--subject", default="sub-01", help="Subject to fetch")
    parser.add_argument(
        "--season",
        type=int,
        default=1,
        help="Friends season to fetch transcripts for",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the existing dataset directory before reinstalling",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if args.force and output.exists():
        shutil.rmtree(output)

    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        run(["datalad", "install", "-r", "-s", REPO_URL, str(output)])

    season_dir = output / "stimuli" / "transcripts" / "friends" / f"s{args.season}"
    h5_path = (
        output
        / "fmri"
        / args.subject
        / "func"
        / f"{args.subject}_task-friends_space-MNI152NLin2009cAsym_"
        "atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
    )

    run(["datalad", "get", str(season_dir), str(h5_path)], cwd=output)

    print(f"Installed Algonauts 2025 slice at: {output}")
    print(f"Fetched transcripts: {season_dir}")
    print(f"Fetched target file: {h5_path}")


if __name__ == "__main__":
    main()
