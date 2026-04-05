from __future__ import annotations

import argparse
from pathlib import Path

from brain_1.training.trainer import TrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the brain-1 parcel head.")
    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
        help="Path to model config YAML",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        train_config_path=Path(args.train_config),
        model_config_path=Path(args.model_config),
    )
    run_training(config)


if __name__ == "__main__":
    main()
