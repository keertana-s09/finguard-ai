"""Model training shared by scripts/train_models.py and the app's first-run setup.

Evaluation protocol (scripts/evaluate_models.py): fit on the TRAIN split only,
measure on the held-out TEST split.
Deployment (this module's train_all): after evaluation, the models are re-fitted
on all labelled synthetic history - the usual practice once the protocol has
been validated. The Isolation Forest contamination level is chosen on the
VALIDATION months only.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import f1_score

from src.anomaly_detector import AnomalyDetector
from src.config import (ANOMALY_MODEL_PATH, CLASSIFIER_PATH, FORECAST_MODEL_PATH, GENERATED_DIR, MODELS_DIR)
from src.expense_classifier import ExpenseClassifier, train_classifier
from src.spending_predictor import SpendingForecaster

CONTAMINATION_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]


def ensure_datasets():
    """Generate the synthetic datasets if they are missing."""
    if not (GENERATED_DIR / "transactions_full.csv").exists():
        from scripts.generate_dataset import main as generate
        generate()


def load_datasets() -> dict:
    ensure_datasets()
    return {
        "clf_train": pd.read_csv(GENERATED_DIR / "classifier_train.csv"),
        "clf_test": pd.read_csv(GENERATED_DIR / "classifier_test.csv"),
        "anomaly": pd.read_csv(GENERATED_DIR / "anomaly_dataset.csv").rename(columns={"is_anomaly": "label"}),
        "monthly": pd.read_csv(GENERATED_DIR / "monthly_totals.csv"),
    }


def select_contamination(anomaly_df: pd.DataFrame) -> tuple[float, list]:
    """Fit on train months, pick the contamination with best F1 on validation months."""
    fit_rows = anomaly_df[anomaly_df["split"] == "train"]
    scores = []
    for c in CONTAMINATION_GRID:
        det = AnomalyDetector(contamination=c).fit(fit_rows, group_col="profile_id")
        scored = det.score(anomaly_df[anomaly_df["split"] != "test"], group_col="profile_id")
        val = scored[scored["split"] == "val"]
        scores.append({"contamination": c, "val_f1": round(f1_score(val["label"], val["is_anomaly"]), 4)})
    best = max(scores, key=lambda r: r["val_f1"])["contamination"]
    return best, scores


def train_all(verbose: bool = True) -> dict:
    data = load_datasets()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. classifier on all labelled expense descriptions
    all_desc = pd.concat([data["clf_train"], data["clf_test"]], ignore_index=True)
    clf = train_classifier(all_desc)
    clf.save(CLASSIFIER_PATH)

    # 2. Isolation Forest: contamination chosen on validation months, then fit on all history
    contamination, grid = select_contamination(data["anomaly"])
    detector = AnomalyDetector(contamination=contamination).fit(data["anomaly"], group_col="profile_id")
    detector.save(ANOMALY_MODEL_PATH)

    # 3. Ridge forecaster on all monthly history
    forecaster = SpendingForecaster().fit(data["monthly"], group_col="profile_id")
    forecaster.save(FORECAST_MODEL_PATH)

    info = {"classifier_rows": len(all_desc), "classifier_classes": clf.classes_,
            "anomaly_rows": len(data["anomaly"]), "anomaly_contamination": contamination,
            "contamination_search": grid, "forecast_profiles": int(data["monthly"]["profile_id"].nunique()),
            "forecast_months": len(data["monthly"])}
    (MODELS_DIR / "training_info.json").write_text(json.dumps(info, indent=2))
    if verbose:
        print(json.dumps(info, indent=2))
    return info


def load_models(auto_train: bool = True) -> dict:
    """Load the three saved models; train them first if missing or incompatible."""
    try:
        return {"classifier": ExpenseClassifier.load(CLASSIFIER_PATH),
                "anomaly": AnomalyDetector.load(ANOMALY_MODEL_PATH),
                "forecaster": SpendingForecaster.load(FORECAST_MODEL_PATH)}
    except Exception:
        if not auto_train:
            raise
        train_all(verbose=False)
        return load_models(auto_train=False)
