from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Join processed text/audio/video manifests into one trimodal manifest.")
    parser.add_argument("--text-manifest", required=True)
    parser.add_argument("--audio-manifest", required=True)
    parser.add_argument("--video-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text_rows = [json.loads(line) for line in Path(args.text_manifest).read_text(encoding="utf-8").splitlines()]
    audio_rows = [json.loads(line) for line in Path(args.audio_manifest).read_text(encoding="utf-8").splitlines()]
    video_rows = [json.loads(line) for line in Path(args.video_manifest).read_text(encoding="utf-8").splitlines()]
    audio_index = {(row["subject_id"], row["stimulus_id"]): row for row in audio_rows}
    video_index = {(row["subject_id"], row["stimulus_id"]): row for row in video_rows}

    merged: list[dict[str, object]] = []
    for text in text_rows:
        key = (text["subject_id"], text["stimulus_id"])
        if key not in audio_index or key not in video_index:
            continue
        audio = audio_index[key]
        video = video_index[key]
        merged.append(
            {
                "dataset_name": "algonauts2025_text_audio_video",
                "subject_id": text["subject_id"],
                "subject_index": text["subject_index"],
                "stimulus_id": text["stimulus_id"],
                "text_feature_path": text["feature_path"],
                "audio_feature_path": audio["feature_path"],
                "video_feature_path": video["feature_path"],
                "target_path": text["target_path"],
                "split": text.get("split", "train"),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row) + "\n")

    print(f"merged_rows={len(merged)}")


if __name__ == "__main__":
    main()
