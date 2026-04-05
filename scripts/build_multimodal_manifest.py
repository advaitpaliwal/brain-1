from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Join processed manifests into a multimodal manifest.")
    parser.add_argument("--text-manifest", required=True, help="Processed text manifest JSONL")
    parser.add_argument("--video-manifest", required=True, help="Processed video manifest JSONL")
    parser.add_argument("--output", required=True, help="Output multimodal manifest JSONL")
    args = parser.parse_args()

    text_manifest = Path(args.text_manifest).expanduser().resolve()
    video_manifest = Path(args.video_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    text_rows = [json.loads(line) for line in text_manifest.read_text(encoding="utf-8").splitlines()]
    video_rows = [json.loads(line) for line in video_manifest.read_text(encoding="utf-8").splitlines()]
    video_index = {(row["subject_id"], row["stimulus_id"]): row for row in video_rows}

    merged: list[dict[str, object]] = []
    for text in text_rows:
        key = (text["subject_id"], text["stimulus_id"])
        if key not in video_index:
            continue
        video = video_index[key]
        merged.append(
            {
                "dataset_name": "algonauts2025_text_video",
                "subject_id": text["subject_id"],
                "subject_index": text["subject_index"],
                "stimulus_id": text["stimulus_id"],
                "text_feature_path": text["feature_path"],
                "video_feature_path": video["feature_path"],
                "target_path": text["target_path"],
                "split": text.get("split", "train"),
            }
        )

    with output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row) + "\n")

    print(f"merged_rows={len(merged)}")


if __name__ == "__main__":
    main()
