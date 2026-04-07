from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import torch


TR_SECONDS = 1.49


def ffmpeg_extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample TR-aligned video frames from Algonauts movies.")
    parser.add_argument("--raw-manifest", required=True, help="Input raw video manifest JSONL")
    parser.add_argument("--frames-root", required=True, help="Output root for sampled frames")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL manifest")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of rows to process")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional max number of frames per stimulus",
    )
    args = parser.parse_args()

    raw_manifest = Path(args.raw_manifest).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in raw_manifest.read_text(encoding="utf-8").splitlines()]
    if args.limit > 0:
        rows = rows[: args.limit]

    processed_rows: list[dict[str, object]] = []
    sampled_stimuli: set[str] = set()

    for row in rows:
        video_path = Path(row["video_path"])
        frame_dir = frames_root / row["stimulus_id"]

        if row["stimulus_id"] not in sampled_stimuli:
            if "target_path" in row:
                target = torch.load(Path(row["target_path"]), map_location="cpu")["target"]
                length = int(target.shape[0])
            else:
                with h5py.File(row["target_h5_path"], "r") as target_handle:
                    target = target_handle[row["target_h5_key"]]
                    length = int(target.shape[0])
            if args.max_frames > 0:
                length = min(length, args.max_frames)

            for index in range(length):
                timestamp = (index + 0.5) * TR_SECONDS
                frame_path = frame_dir / f"{index:04d}.png"
                if frame_path.exists():
                    continue
                ffmpeg_extract_frame(video_path, timestamp, frame_path)

            sampled_stimuli.add(row["stimulus_id"])

        processed_rows.append(
            {
                "dataset_name": row["dataset_name"],
                "subject_id": row["subject_id"],
                "subject_index": row["subject_index"],
                "season": row["season"],
                "stimulus_id": row["stimulus_id"],
                "frame_dir": str(frame_dir),
                "target_h5_path": row.get("target_h5_path"),
                "target_h5_key": row.get("target_h5_key"),
                "target_path": row.get("target_path"),
                "split": row.get("split", "train"),
            }
        )
        print(f"prepared frames for stimulus={row['stimulus_id']} subject={row['subject_id']}")

    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in processed_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote frame manifest to {output_manifest}")


if __name__ == "__main__":
    main()
