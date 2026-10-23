"""Anomaly detection on expense transactions with Isolation Forest.

Features (per expense)
----------------------
* log_amount            - log(1 + amount). The raw amount is heavy-tailed, so
                          the model sees its log (raw amount is shown to the user).
* excess_over_typical   - max(0, log(amount / typical amount for the category)).
                          The typical amount is the median of the user's PREVIOUS
                          transactions in that category (no look-ahead). This is
                          how the expense CATEGORY enters the model. It is
                          one-sided because only unusually HIGH spending matters
                          for budgeting.
* day_of_week           - 0 = Monday ... 6 = Sunday

Optional (ablation only): one-hot category columns. scripts/evaluate_models.py
shows that adding them lowers validation F1, so they are off by default.

Transaction IDs are never used as features. Income rows are excluded.

The model produces only a score and a flag. The human-readable explanation is
built separately from simple category statistics and is labelled as such.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import (ANOMALY_MODEL_PATH, EXPENSE_CATEGORIES, HIGH_SIDE_RATIO, MIN_CATEGORY_HISTORY,
                        MIN_EXPENSES_FOR_ANOMALY, RANDOM_SEED)
from src.utils import format_inr

HISTORY_WINDOW = 60   # previous transactions used for the 'typical amount'


def add_history_features(expenses: pd.DataFrame, reference_medians: dict, group_col: str | None = None) -> pd.DataFrame:
    """Add typical_amount / history_count using only PAST transactions.

    When a category has fewer than MIN_CATEGORY_HISTORY earlier transactions the
    typical amount falls back to the reference median learned from training data.
    """
    df = expenses.copy()
    df["_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["_dt"], kind="stable")
    keys = ([group_col] if group_col else []) + ["category"]
    grouped = df.groupby(keys, sort=False)["amount"]
    df["history_count"] = grouped.cumcount()
    df["typical_amount"] = grouped.transform(
        lambda s: s.shift(1).rolling(HISTORY_WINDOW, min_periods=MIN_CATEGORY_HISTORY).median())
    df["typical_source"] = np.where(df["typical_amount"].notna(), "your history", "reference data")
    fallback = df["category"].map(reference_medians).astype(float)
    df["typical_amount"] = df["typical_amount"].fillna(fallback).fillna(float(np.median(list(reference_medians.values()))))
    return df.drop(columns="_dt").sort_index()


def build_feature_matrix(df: pd.DataFrame, use_category_onehot: bool = False) -> pd.DataFrame:
    dt = pd.to_datetime(df["date"])
    X = pd.DataFrame(index=df.index)
    X["log_amount"] = np.log1p(df["amount"].astype(float))
    ratio = df["amount"].astype(float) / df["typical_amount"].astype(float)
    X["excess_over_typical"] = np.log(ratio).clip(lower=0)
    X["day_of_week"] = dt.dt.day_of_week
    if use_category_onehot:
        for cat in EXPENSE_CATEGORIES:
            X[f"cat_{cat}"] = (df["category"] == cat).astype(int)
    return X


def explain(row) -> str:
    """Interpretable statistical context for a flagged transaction.

    NOTE: this text comes from comparing the amount with the category's typical
    amount; Isolation Forest itself does not produce explanations.
    """
    amount, typical, cat = row["amount"], row["typical_amount"], row["category"]
    ratio = amount / typical if typical else np.nan
    basis = ("your previous " + cat.lower() + " transactions" if row["typical_source"] == "your history"
             else "typical " + cat.lower() + " spending in the reference data (not enough of your own history yet)")
    if ratio >= HIGH_SIDE_RATIO:
        return (f"Your {format_inr(amount)} {cat.lower()} transaction is {ratio:.1f}x the typical amount "
                f"({format_inr(typical)}, median of {basis}).")
    if ratio <= 1 / HIGH_SIDE_RATIO:
        return (f"Your {format_inr(amount)} {cat.lower()} transaction is unusually small compared with the "
                f"typical {format_inr(typical)} ({basis}).")
    day = pd.Timestamp(row["date"]).day_name()
    return (f"Amount {format_inr(amount)} is close to the typical {format_inr(typical)} for {cat.lower()}, "
            f"but the combination of category, amount and day ({day}) is uncommon in the training data.")


class AnomalyDetector:
    def __init__(self, contamination: float = 0.02, n_estimators: int = 300, use_category_onehot: bool = False):
        self.contamination = contamination
        self.use_category_onehot = use_category_onehot
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                     random_state=RANDOM_SEED)
        self.reference_medians: dict = {}
        self.feature_names: list = []

    # ------------------------------------------------------------ training
    def fit(self, expenses: pd.DataFrame, group_col: str | None = None):
        """expenses: historical expense rows (date, amount, category[, group_col])."""
        self.reference_medians = expenses.groupby("category")["amount"].median().to_dict()
        feats = add_history_features(expenses, self.reference_medians, group_col)
        X = build_feature_matrix(feats, self.use_category_onehot)
        self.feature_names = list(X.columns)
        self.model.fit(X)
        return self

    # ------------------------------------------------------------ scoring
    def score(self, expenses: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
        """Return the input rows plus anomaly_score (higher = more unusual),
        is_anomaly, typical_amount, ratio_to_typical and explanation."""
        feats = add_history_features(expenses, self.reference_medians, group_col)
        X = build_feature_matrix(feats, self.use_category_onehot)[self.feature_names]
        # decision_function: negative = anomalous. Flip the sign so higher = more unusual.
        feats["anomaly_score"] = (-self.model.decision_function(X)).round(4)
        feats["is_anomaly"] = self.model.predict(X) == -1
        feats["ratio_to_typical"] = (feats["amount"] / feats["typical_amount"]).round(2)
        feats["anomaly_explanation"] = [explain(r) if r["is_anomaly"] else "" for _, r in feats.iterrows()]
        return feats

    def detect(self, transactions: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """App entry point: filters expenses, handles small datasets gracefully.

        Returns (scored_expenses, status_message).
        """
        expenses = transactions[(transactions["type"] == "Expense") & transactions["category"].notna()]
        if len(expenses) < MIN_EXPENSES_FOR_ANOMALY:
            return expenses.iloc[0:0].assign(anomaly_score=[], is_anomaly=[], typical_amount=[],
                                             ratio_to_typical=[], anomaly_explanation=[]), (
                f"Anomaly detection skipped: only {len(expenses)} categorised expense(s); "
                f"at least {MIN_EXPENSES_FOR_ANOMALY} are needed.")
        scored = self.score(expenses)
        n_fallback = int((scored["typical_source"] == "reference data").sum())
        msg = f"Scored {len(scored)} expenses; {int(scored['is_anomaly'].sum())} flagged as unusual."
        if n_fallback:
            msg += (f" {n_fallback} transaction(s) had little category history, so reference-data "
                    f"medians were used as the typical amount.")
        return scored, msg

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path = ANOMALY_MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path = ANOMALY_MODEL_PATH) -> "AnomalyDetector":
        return joblib.load(path)


def high_side_anomalies(scored: pd.DataFrame) -> pd.DataFrame:
    """Flagged transactions that are unusually HIGH (>= HIGH_SIDE_RATIO x typical).
    These are treated as one-off spending by the forecaster and the agent."""
    if scored is None or scored.empty:
        return scored
    return scored[scored["is_anomaly"] & (scored["amount"] >= HIGH_SIDE_RATIO * scored["typical_amount"])]


def rule_baseline(scored: pd.DataFrame, ratio: float = 3.0) -> pd.Series:
    """Baseline for evaluation: flag any expense > `ratio` x its typical amount."""
    return scored["amount"] / scored["typical_amount"] > ratio
