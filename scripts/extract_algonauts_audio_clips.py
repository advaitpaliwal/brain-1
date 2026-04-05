from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py


TR_SECONDS = 1.49


def ffmpeg_extract_audio(video_path: Path, start: float, duration: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TR-aligned audio clips from Algonauts videos.")
    parser.add_argument("--raw-manifest", required=True, help="Input raw video manifest JSONL")
    parser.add_argument("--clips-root", required=True, help="Output root for WAV clips")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL manifest")
    parser.add_argument("--max-clips", type=int, default=0, help="Optional max clips per stimulus")
    args = parser.parse_args()

    raw_manifest = Path(args.raw_manifest).expanduser().resolve()
    clips_root = Path(args.clips_root).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    clips_root.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in raw_manifest.read_text(encoding="utf-8").splitlines()]
    processed_rows: list[dict[str, object]] = []
    processed_stimuli: set[str] = set()

    for row in rows:
        video_path = Path(row["video_path"])
        clip_dir = clips_root / row["stimulus_id"]

        if row["stimulus_id"] not in processed_stimuli:
            with h5py.File(row["target_h5_path"], "r") as target_handle:
                length = int(target_handle[row["target_h5_key"]].shape[0])
            if args.max_clips > 0:
                length = min(length, args.max_clips)

            for index in range(length):
                clip_path = clip_dir / f"{index:04d}.wav"
                if clip_path.exists():
                    continue
                ffmpeg_extract_audio(video_path, start=index * TR_SECONDS, duration=TR_SECONDS, output_path=clip_path)

            processed_stimuli.add(row["stimulus_id"])

        processed_rows.append(
            {
                "dataset_name": "algonauts2025_audio",
                "subject_id": row["subject_id"],
                "subject_index": row["subject_index"],
                "season": row["season"],
                "stimulus_id": row["stimulus_id"],
                "clip_dir": str(clip_dir),
                "target_h5_path": row["target_h5_path"],
                "target_h5_key": row["target_h5_key"],
            }
        )
        print(f"prepared audio clips for stimulus={row['stimulus_id']} subject={row['subject_id']}")

    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in processed_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote audio clip manifest to {output_manifest}")


if __name__ == "__main__":
    main()
