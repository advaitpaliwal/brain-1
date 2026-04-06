from __future__ import annotations

import argparse
import json
from pathlib import Path

from brain_1.utils import parse_interval_tier, words_per_tr


TR_SECONDS = 2.0


def parse_csv(value: str) -> list[str]:
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw Lebel 2023 text manifest.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the ds003020 OpenNeuro checkout",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Optional comma-separated subject ids, e.g. UTS01,UTS02",
    )
    parser.add_argument(
        "--stories",
        default="",
        help="Optional comma-separated story ids, e.g. wheretheressmoke,legacy",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of manifest rows to include",
    )
    parser.add_argument(
        "--subject-index-offset",
        type=int,
        default=100,
        help="Base offset used to keep Lebel subject indices disjoint from Algonauts",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/lebel2023_text_raw.jsonl",
        help="Output JSONL manifest path",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    respdict_path = root / "derivatives" / "respdict.json"
    respdict = json.loads(respdict_path.read_text(encoding="utf-8"))

    available_subjects = sorted(
        path.name
        for path in (root / "derivatives" / "preprocessed_data").iterdir()
        if path.is_dir()
    )
    subjects = parse_csv(args.subjects) if args.subjects else available_subjects

    available_stories = sorted(respdict)
    stories = parse_csv(args.stories) if args.stories else available_stories

    rows: list[dict[str, object]] = []
    subject_indices = {
        subject: args.subject_index_offset + index
        for index, subject in enumerate(sorted(subjects))
    }

    for subject in subjects:
        for story in stories:
            textgrid_path = root / "derivatives" / "TextGrids" / f"{story}.TextGrid"
            target_path = root / "derivatives" / "preprocessed_data" / subject / f"{story}.hf5"
            if not textgrid_path.exists():
                print(f"skipping missing TextGrid: {textgrid_path}")
                continue
            if not target_path.exists():
                print(f"skipping missing target: {target_path}")
                continue

            intervals = parse_interval_tier(textgrid_path, "word")
            texts = words_per_tr(intervals, tr_count=int(respdict[story]), tr_seconds=TR_SECONDS)
            split = "test" if story == "wheretheressmoke" else "train"
            rows.append(
                {
                    "dataset_name": "lebel2023",
                    "subject_id": subject,
                    "subject_index": subject_indices[subject],
                    "split": split,
                    "stimulus_id": story,
                    "texts": texts,
                    "target_h5_path": str(target_path),
                    "target_h5_key": "data",
                    "tr_seconds": TR_SECONDS,
                }
            )
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} Lebel manifest rows to {output}")


if __name__ == "__main__":
    main()
