from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def normalize_stimulus_id(raw_value: object) -> str:
    text = str(raw_value)
    text = Path(text).stem
    if text.startswith("vid") and text[3:].isdigit():
        text = text[3:]
    return text


def joined_captions(caption_payload: dict[str, list[str]]) -> str:
    captions = next(iter(caption_payload.values()))
    return " ".join(str(caption).strip() for caption in captions if str(caption).strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a text-style raw manifest for BOLD Moments from prepared betas and captions."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the ds005165 OpenNeuro checkout")
    parser.add_argument("--subject", required=True, help="Subject id, e.g. sub-01")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Prepared beta split to use",
    )
    parser.add_argument(
        "--hemi",
        default="left",
        choices=["left", "right"],
        help="Hemisphere prepared beta file to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of stimuli to include",
    )
    parser.add_argument(
        "--subject-index-offset",
        type=int,
        default=200,
        help="Base offset used to keep BOLD Moments subject indices disjoint",
    )
    parser.add_argument(
        "--target-root",
        default="data/processed/bold_moments_targets",
        help="Directory for saved target tensors",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/bold_moments_text_raw.jsonl",
        help="Output JSONL manifest path",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    subject_number = int(args.subject.split("-")[-1])
    subject_index = args.subject_index_offset + subject_number - 1

    captions_path = root / "derivatives" / "stimuli_metadata" / "llm_frame_annotations.json"
    captions = json.loads(captions_path.read_text(encoding="utf-8"))

    pkl_path = (
        root
        / "derivatives"
        / "versionB"
        / "fsaverage"
        / "GLM"
        / args.subject
        / "prepared_betas"
        / f"{args.subject}_organized_betas_task-{args.split}_hemi-{args.hemi}_normalized.pkl"
    )
    with pkl_path.open("rb") as handle:
        prepared_betas = pickle.load(handle)
    betas, stims = prepared_betas[0], prepared_betas[1]
    if not isinstance(betas, np.ndarray):
        betas = np.asarray(betas)
    mean_betas = betas.mean(axis=1)

    rows: list[dict[str, object]] = []
    for index, stim in enumerate(stims):
        stimulus_id = normalize_stimulus_id(stim)
        if stimulus_id not in captions:
            print(f"skipping stimulus with missing captions: {stimulus_id}")
            continue

        target_path = target_root / f"{args.subject}_{args.split}_{args.hemi}_{stimulus_id}_target.pt"
        target = torch.tensor(mean_betas[index], dtype=torch.float32).unsqueeze(0)
        torch.save({"target": target}, target_path)

        rows.append(
            {
                "dataset_name": f"bold_moments_{args.hemi}",
                "subject_id": args.subject,
                "subject_index": subject_index,
                "split": args.split,
                "stimulus_id": f"{args.split}_{stimulus_id}",
                "texts": [joined_captions(captions[stimulus_id])],
                "target_path": str(target_path),
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(
        f"Wrote {len(rows)} BOLD Moments rows to {output} "
        f"with target dim {mean_betas.shape[-1]}"
    )


if __name__ == "__main__":
    main()
