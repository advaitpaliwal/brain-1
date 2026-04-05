from pathlib import Path

import yaml

from brain_1.training.trainer import TrainingConfig, run_training


def test_run_training_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = repo_root / "tests" / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_config = {
        "seed": 1337,
        "data": {
            "manifest_path": str(tmp_dir / "missing.jsonl"),
            "synthetic_fallback_size": 4,
            "synthetic_seq_len": 8,
        },
        "optimization": {
            "batch_size": 2,
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "max_steps": 2,
            "warmup_steps": 0,
        },
        "loss": {
            "mse_weight": 1.0,
            "correlation_weight": 0.1,
        },
        "training": {
            "mixed_precision": "no",
            "grad_clip_norm": 1.0,
            "log_every": 1,
            "eval_every": 100,
            "checkpoint_every": 100,
        },
    }
    model_config = {
        "backbone": {
            "name": "qwen2_5_omni_7b",
            "pretrained_id": "Qwen/Qwen2.5-Omni-7B",
            "freeze_backbone": True,
            "input_feature_dim": 32,
            "hidden_size": 32,
        },
        "projection": {
            "hidden_size": 16,
            "dropout": 0.1,
        },
        "temporal_adapter": {
            "type": "transformer",
            "layers": 1,
            "heads": 4,
            "dropout": 0.1,
        },
        "hrf": {
            "kernel_size": 3,
            "learnable": True,
        },
        "head": {
            "output_dim": 12,
            "subject_embedding_dim": 8,
        },
    }

    train_path = tmp_dir / "train.yaml"
    model_path = tmp_dir / "model.yaml"
    train_path.write_text(yaml.safe_dump(train_config), encoding="utf-8")
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")

    run_training(
        TrainingConfig(
            train_config_path=train_path,
            model_config_path=model_path,
        )
    )
