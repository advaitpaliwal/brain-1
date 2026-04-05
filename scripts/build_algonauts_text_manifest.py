from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py


SEGMENT_RE = re.compile(r"friends_(s\d{2}e\d{2}[a-z])\.tsv$")


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


def choose_h5_key(keys: list[str], segment_id: str) -> str:
    expected_suffix = f"task-{segment_id}"
    matches = [key for key in keys if key.endswith(expected_suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one H5 key for {segment_id}, got {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw Algonauts text manifest.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the Algonauts 2025 DataLad dataset root",
    )
    parser.add_argument("--subject", default="sub-01", help="Subject id, e.g. sub-01")
    parser.add_argument(
        "--subjects",
        default="",
        help="Optional comma-separated list of subject ids; overrides --subject",
    )
    parser.add_argument("--season", type=int, default=1, help="Friends season number")
    parser.add_argument(
        "--seasons",
        default="",
        help="Optional season list or range, e.g. 1-6 or 1,2,6; overrides --season",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of transcript segments to include",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/algonauts2025_text_raw.jsonl",
        help="Output JSONL manifest path",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    subjects = (
        [value.strip() for value in args.subjects.split(",") if value.strip()]
        if args.subjects
        else [args.subject]
    )
    seasons = parse_seasons(args.seasons) if args.seasons else [args.season]

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
            transcript_dir = root / "stimuli" / "transcripts" / "friends" / f"s{season}"
            transcript_paths = sorted(transcript_dir.glob("friends_*.tsv"))
            if args.limit > 0:
                transcript_paths = transcript_paths[: args.limit]

            for transcript_path in transcript_paths:
                match = SEGMENT_RE.search(transcript_path.name)
                if not match:
                    continue
                segment_id = match.group(1)
                try:
                    h5_key = choose_h5_key(keys, segment_id)
                except ValueError:
                    print(f"skipping missing target subject={subject} stimulus={segment_id}")
                    continue
                rows.append(
                    {
                        "dataset_name": "algonauts2025",
                        "subject_id": subject,
                        "subject_index": subject_index,
                        "split": "train" if season < 7 else "test",
                        "season": season,
                        "stimulus_id": segment_id,
                        "transcript_path": str(transcript_path),
                        "target_h5_path": str(h5_path),
                        "target_h5_key": h5_key,
                    }
                )

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} manifest rows to {output}")


if __name__ == "__main__":
    main()
