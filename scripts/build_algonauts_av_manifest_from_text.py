from __future__ import annotations

import argparse
import json
from pathlib import Path


def season_dir_from_stimulus(stimulus_id: str) -> str:
    return f"s{int(stimulus_id[1:3])}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an Algonauts audio/video raw manifest from a processed text manifest."
    )
    parser.add_argument("--text-manifest", required=True, help="Processed text manifest JSONL")
    parser.add_argument("--video-root", required=True, help="Root directory containing Friends videos")
    parser.add_argument("--output", required=True, help="Output raw AV manifest JSONL")
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Skip rows whose expected video file does not exist",
    )
    args = parser.parse_args()

    text_manifest = Path(args.text_manifest).expanduser().resolve()
    video_root = Path(args.video_root)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in text_manifest.read_text(encoding="utf-8").splitlines()]
    output_rows: list[dict[str, object]] = []

    for row in rows:
        stimulus_id = row["stimulus_id"]
        video_path = video_root / season_dir_from_stimulus(stimulus_id) / f"friends_{stimulus_id}.mkv"
        if args.skip_missing_videos and not video_path.exists():
            continue
        output_rows.append(
            {
                "dataset_name": "algonauts2025_av",
                "subject_id": row["subject_id"],
                "subject_index": row["subject_index"],
                "stimulus_id": stimulus_id,
                "season": int(stimulus_id[1:3]),
                "video_path": str(video_path),
                "target_path": row["target_path"],
                "split": row.get("split", "train"),
            }
        )

    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(output_rows)} AV manifest rows to {output}")


if __name__ == "__main__":
    main()
