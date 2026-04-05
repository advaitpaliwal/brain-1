from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def encode_images(images, processor, model, device: torch.device) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if hasattr(model, "get_image_features"):
        features = model.get_image_features(**inputs)
        if hasattr(features, "image_embeds"):
            features = features.image_embeds
        elif hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif not isinstance(features, torch.Tensor):
            features = features[0]
    else:
        outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            features = hidden[:, 0]
    return features.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frame features from sampled Algonauts video frames.")
    parser.add_argument("--frame-manifest", required=True, help="Input frame manifest JSONL")
    parser.add_argument("--output-manifest", required=True, help="Output processed manifest JSONL")
    parser.add_argument("--feature-root", required=True, help="Directory for saved features/targets")
    parser.add_argument(
        "--model-id",
        default="google/siglip-base-patch16-224",
        help="Image encoder model id",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Image encoding batch size")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or mps")
    parser.add_argument(
        "--share-features-across-subjects",
        action="store_true",
        help="Store one feature tensor per stimulus",
    )
    args = parser.parse_args()

    frame_manifest = Path(args.frame_manifest).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in frame_manifest.read_text(encoding="utf-8").splitlines()]
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    processed_rows: list[dict[str, object]] = []
    for row in rows:
        feature_stem = row["stimulus_id"] if args.share_features_across_subjects else f"{row['subject_id']}_{row['stimulus_id']}"
        feature_path = feature_root / f"{feature_stem}_features.pt"
        target_path = feature_root / f"{row['subject_id']}_{row['stimulus_id']}_target.pt"

        if not feature_path.exists():
            frame_paths = sorted(Path(row["frame_dir"]).glob("*.png"))
            features: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(frame_paths), args.batch_size):
                    images = [Image.open(path).convert("RGB") for path in frame_paths[start : start + args.batch_size]]
                    features.append(encode_images(images, processor, model, device))
            feature_tensor = torch.cat(features, dim=0)
            torch.save({"features": feature_tensor}, feature_path)
        else:
            feature_tensor = torch.load(feature_path, map_location="cpu")["features"]

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
                "split": "train" if int(row["season"]) < 7 else "test",
            }
        )
        print(f"processed video stimulus={row['stimulus_id']} subject={row['subject_id']}")

    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in processed_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote processed video manifest to {output_manifest}")


if __name__ == "__main__":
    main()
