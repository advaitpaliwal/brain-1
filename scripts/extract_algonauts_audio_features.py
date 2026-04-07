from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel


def encode_audio(wavs, feature_extractor, model, device: torch.device) -> torch.Tensor:
    inputs = feature_extractor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    features = hidden.mean(dim=1)
    return features.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Algonauts audio features with w2v-BERT 2.0.")
    parser.add_argument("--clip-manifest", required=True, help="Input audio clip manifest JSONL")
    parser.add_argument("--output-manifest", required=True, help="Output processed manifest JSONL")
    parser.add_argument("--feature-root", required=True, help="Directory for saved features/targets")
    parser.add_argument(
        "--model-id",
        default="facebook/w2v-bert-2.0",
        help="Audio encoder checkpoint id",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Audio encoding batch size")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or mps")
    parser.add_argument(
        "--share-features-across-subjects",
        action="store_true",
        help="Store one feature tensor per stimulus",
    )
    args = parser.parse_args()

    clip_manifest = Path(args.clip_manifest).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in clip_manifest.read_text(encoding="utf-8").splitlines()]
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = Wav2Vec2BertModel.from_pretrained(args.model_id)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    processed_rows: list[dict[str, object]] = []
    for row in rows:
        feature_stem = row["stimulus_id"] if args.share_features_across_subjects else f"{row['subject_id']}_{row['stimulus_id']}"
        feature_path = feature_root / f"{feature_stem}_features.pt"
        target_path = feature_root / f"{row['subject_id']}_{row['stimulus_id']}_target.pt"

        if not feature_path.exists():
            clip_paths = sorted(Path(row["clip_dir"]).glob("*.wav"))
            features: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(clip_paths), args.batch_size):
                    wavs = []
                    for path in clip_paths[start : start + args.batch_size]:
                        wav, sr = sf.read(path)
                        if sr != 16000:
                            raise ValueError(f"Expected 16000 Hz audio, got {sr}")
                        wavs.append(wav)
                    features.append(encode_audio(wavs, feature_extractor, model, device))
            feature_tensor = torch.cat(features, dim=0)
            torch.save({"features": feature_tensor}, feature_path)
        else:
            feature_tensor = torch.load(feature_path, map_location="cpu")["features"]

        if row.get("target_path"):
            target = torch.load(Path(row["target_path"]), map_location="cpu")["target"].float()
        else:
            with h5py.File(row["target_h5_path"], "r") as target_handle:
                target = torch.tensor(target_handle[row["target_h5_key"]][:], dtype=torch.float32)

        length = min(feature_tensor.shape[0], target.shape[0])
        target = target[:length]
        torch.save({"target": target}, target_path)

        processed_rows.append(
            {
                "dataset_name": row["dataset_name"],
                "subject_id": row["subject_id"],
                "subject_index": row["subject_index"],
                "stimulus_id": row["stimulus_id"],
                "feature_path": str(feature_path),
                "target_path": str(target_path),
                "split": row.get("split", ("train" if int(row["season"]) < 7 else "test")),
            }
        )
        print(f"processed audio stimulus={row['stimulus_id']} subject={row['subject_id']}")

    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in processed_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote processed audio manifest to {output_manifest}")


if __name__ == "__main__":
    main()
