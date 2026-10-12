"""Train and save the three models used by the app.

Run from the project root:
    python scripts/train_models.py

Creates models/expense_classifier.joblib, models/anomaly_detector.joblib,
models/spending_forecaster.joblib and models/training_info.json.
(The datasets are generated automatically if data/generated/ is empty.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training import train_all  # noqa: E402

if __name__ == "__main__":
    train_all(verbose=True)
    print("Models saved to the models/ folder.")
