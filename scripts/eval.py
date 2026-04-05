from __future__ import annotations

from brain_1.training.metrics import regression_metrics


def main() -> None:
    metrics = regression_metrics(pred=None, target=None)
    print("Stub evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
