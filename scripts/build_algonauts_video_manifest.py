from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from build_algonauts_text_manifest import choose_h5_key, parse_seasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw Algonauts video manifest.")
    parser.add_argument("--dataset-root", required=True, help="Algonauts DataLad dataset root")
    parser.add_argument("--subjects", required=True, help="Comma-separated subject ids")
    parser.add_argument(
        "--seasons",
        default="1",
        help="Season list or range, e.g. 1 or 1-6",
    )
    parser.add_argument(
        "--limit-per-season",
        type=int,
        default=0,
        help="Optional max number of movie segments per season",
    )
    parser.add_argument("--output", required=True, help="Output JSONL manifest path")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    seasons = parse_seasons(args.seasons)
    rows: list[dict[str, object]] = []

    for subject in subjects:
        h5_path = (
            root
            / "fmri"
            / subject
            / "func"
            / f"{subject}_task-friends_space-MNI152NLin2009cAsym_"
            "atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
        )
        with h5py.File(h5_path, "r") as handle:
            keys = list(handle.keys())

        subject_index = int(subject.split("-")[-1]) - 1

        for season in seasons:
            season_dir = root / "stimuli" / "movies" / "friends" / f"s{season}"
            movie_files = sorted(season_dir.glob("friends_*.mkv"))
            if args.limit_per_season > 0:
                movie_files = movie_files[: args.limit_per_season]

            for movie_file in movie_files:
                stem = movie_file.stem.removeprefix("friends_")
                try:
                    h5_key = choose_h5_key(keys, stem)
                except ValueError:
                    print(f"skipping missing target subject={subject} stimulus={stem}")
                    continue
                rows.append(
                    {
                        "dataset_name": "algonauts2025_video",
                        "subject_id": subject,
                        "subject_index": subject_index,
                        "season": season,
                        "stimulus_id": stem,
                        "video_path": str(movie_file),
                        "target_h5_path": str(h5_path),
                        "target_h5_key": h5_key,
                    }
                )

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} video manifest rows to {output}")


if __name__ == "__main__":
    main()
