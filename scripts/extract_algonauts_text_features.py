from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = (
                outputs.last_hidden_state
                if hasattr(outputs, "last_hidden_state")
                else outputs[0]
            )
            pooled = mean_pool(hidden, encoded["attention_mask"])
            features.append(pooled.cpu())
    return torch.cat(features, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text features from Algonauts transcript chunks.")
    parser.add_argument("--raw-manifest", required=True, help="Input raw JSONL manifest")
    parser.add_argument(
        "--output-manifest",
        default="data/manifests/algonauts2025_text_train.jsonl",
        help="Output processed JSONL manifest",
    )
    parser.add_argument(
        "--feature-root",
        default="data/processed/algonauts2025_text",
        help="Directory for saved feature and target tensors",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id for text feature extraction",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of manifest rows to process",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Text encoding batch size")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if outputs already exist",
    )
    parser.add_argument(
        "--share-features-across-subjects",
        action="store_true",
        help="Store one feature tensor per stimulus instead of one per subject/stimulus pair",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer and AutoModel",
    )
    args = parser.parse_args()

    raw_manifest = Path(args.raw_manifest).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    device = torch.device(args.device)
    model.to(device)

    processed_rows: list[dict[str, object]] = []
    with raw_manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if args.limit > 0:
        rows = rows[: args.limit]

    text_cache: dict[str, torch.Tensor] = {}

    for row in rows:
        feature_stem = (
            f"{row['stimulus_id']}"
            if args.share_features_across_subjects
            else f"{row['subject_id']}_{row['stimulus_id']}"
        )
        feature_path = feature_root / f"{feature_stem}_features.pt"
        target_path = feature_root / f"{row['subject_id']}_{row['stimulus_id']}_target.pt"
        if feature_path.exists() and target_path.exists() and not args.overwrite:
            processed_rows.append(
                {
                    "dataset_name": row["dataset_name"],
                    "subject_id": row["subject_id"],
                    "subject_index": row["subject_index"],
                    "stimulus_id": row["stimulus_id"],
                    "feature_path": str(feature_path),
                    "target_path": str(target_path),
                    "split": row["split"],
                }
            )
            print(f"skipped stimulus={row['stimulus_id']} outputs already exist")
            continue

        if "texts" in row:
            texts = [str(text) for text in row["texts"]]
        else:
            transcript_path = Path(row["transcript_path"])
            df = pd.read_csv(transcript_path, sep="\t")
            texts = df["text_per_tr"].fillna("").astype(str).tolist()

        uncached = [text for text in texts if text not in text_cache]
        if uncached:
            unique_uncached = list(dict.fromkeys(uncached))
            uncached_features = encode_texts(
                unique_uncached,
                tokenizer,
                model,
                args.batch_size,
                device,
            )
            for text, feature in zip(unique_uncached, uncached_features, strict=True):
                text_cache[text] = feature

        features = torch.stack([text_cache[text] for text in texts]).float()

        if "target_path" in row:
            target = torch.load(Path(row["target_path"]), map_location="cpu")["target"].float()
        else:
            with h5py.File(row["target_h5_path"], "r") as target_handle:
                target = torch.tensor(target_handle[row["target_h5_key"]][:], dtype=torch.float32)

        length = min(features.shape[0], target.shape[0])
        features = features[:length]
        target = target[:length]

        torch.save({"features": features}, feature_path)
        torch.save({"target": target}, target_path)

        processed_rows.append(
            {
                "dataset_name": row["dataset_name"],
                "subject_id": row["subject_id"],
                "subject_index": row["subject_index"],
                "stimulus_id": row["stimulus_id"],
                "feature_path": str(feature_path),
                "target_path": str(target_path),
                "split": row["split"],
            }
        )
        print(
            f"processed stimulus={row['stimulus_id']} length={length} "
            f"feature_dim={features.shape[-1]}"
        )

    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in processed_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote processed manifest to {output_manifest}")


if __name__ == "__main__":
    main()
