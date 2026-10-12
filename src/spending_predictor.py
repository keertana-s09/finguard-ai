"""Next-month total expense forecast with Ridge Regression.

Features for predicting month t (all from months BEFORE t - no leakage):
    hist_mean             average of ALL months before t (the user's usual level)
    lag_k_rel  (k=1..3)   expense of month t-k divided by hist_mean
    mean_3_rel            average of the last three months divided by hist_mean
    month_sin, month_cos  calendar month of t (captures festive-season peaks)

Target: y / hist_mean. The prediction is multiplied back by hist_mean.
Working with ratios makes the model scale-free, so one model trained on
several synthetic profiles (student ... family) can be applied to any user.

What is forecast: REGULAR monthly spending. One-off transactions that the
anomaly detector flagged as unusually high are excluded from the inputs,
because a single ₹38,000 purchase should not make the model expect another
one next month. (The agent handles those one-offs separately.)

Fallback: when fewer than FORECAST_N_LAGS complete months exist, a clearly
labelled rule-based forecast (average of the available months) is returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FORECAST_MODEL_PATH, FORECAST_N_LAGS
from src.preprocessing import last_month_is_partial, monthly_expense_totals

FEATURES = ["lag_1_rel", "lag_2_rel", "lag_3_rel", "mean_3_rel", "month_sin", "month_cos"]


def make_lag_features(series: pd.Series, months: pd.Series) -> pd.DataFrame:
    """series: monthly totals in chronological order; months: matching pandas Periods.
    Returns one row per month that has 3 previous months, with the target 'y'."""
    s = pd.Series(np.asarray(series, dtype=float))
    m = pd.Series(list(months))
    df = pd.DataFrame({"month": m, "y": s})
    for k in range(1, FORECAST_N_LAGS + 1):
        df[f"lag_{k}"] = s.shift(k)
    df["mean_3"] = df[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
    df["hist_mean"] = s.shift(1).expanding().mean()
    for col in ["lag_1", "lag_2", "lag_3", "mean_3"]:
        df[f"{col}_rel"] = df[col] / df["hist_mean"]
    month_num = m.map(lambda p: p.month).astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    df["month_cos"] = np.cos(2 * np.pi * month_num / 12)
    return df.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)


def features_for_next(series, last_month: pd.Period) -> pd.DataFrame:
    """Feature row for the month AFTER the last observed month."""
    values = np.asarray(series, dtype=float)
    extended = np.append(values, np.nan)  # placeholder target
    months = pd.period_range(end=last_month + 1, periods=len(extended), freq="M")
    return make_lag_features(extended, months).iloc[[-1]].reset_index(drop=True)


@dataclass
class ForecastResult:
    target_month: str
    predicted: float | None
    method: str                       # 'ml_ridge' | 'rule_average' | 'none'
    note: str
    history: pd.DataFrame = field(default_factory=pd.DataFrame)  # month, actual, predicted

    def to_dict(self):
        return {"target_month": self.target_month, "predicted": self.predicted, "method": self.method,
                "note": self.note,
                "history": self.history.assign(month=self.history["month"].astype(str)).to_dict("records")
                if not self.history.empty else []}


class SpendingForecaster:
    def __init__(self, alpha: float = 1.0):
        self.pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

    def fit(self, monthly: pd.DataFrame, group_col: str = "profile_id", value_col: str = "regular_expense"):
        """monthly: rows with group_col, month (YYYY-MM) and value_col. Lag
        features are built inside each group so profiles never mix."""
        X = self.build_training_rows(monthly, group_col, value_col)
        self.pipeline.fit(X[FEATURES], X["y"] / X["hist_mean"])
        return self

    @staticmethod
    def build_training_rows(monthly: pd.DataFrame, group_col: str = "profile_id",
                            value_col: str = "regular_expense") -> pd.DataFrame:
        frames = []
        for gid, g in monthly.sort_values([group_col, "month"]).groupby(group_col):
            feats = make_lag_features(g[value_col], pd.PeriodIndex(g["month"], freq="M"))
            feats[group_col] = gid
            frames.append(feats)
        return pd.concat(frames, ignore_index=True)

    def predict_rows(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(0.0, self.pipeline.predict(X[FEATURES]) * X["hist_mean"].to_numpy())

    # ------------------------------------------------------------ app entry point
    def forecast(self, transactions: pd.DataFrame, one_off_ids=()) -> ForecastResult:
        """Forecast regular spending for the month after the last COMPLETE month.

        one_off_ids: database ids of high-side anomalies to leave out of the inputs.
        """
        all_monthly = monthly_expense_totals(transactions)
        if all_monthly.empty or all_monthly["expense"].sum() == 0:
            return ForecastResult("-", None, "none", "No expense data available for forecasting.")
        one_off_ids = set(int(i) for i in one_off_ids)
        regular_tx = transactions[~transactions["id"].isin(one_off_ids)] if one_off_ids else transactions
        monthly = monthly_expense_totals(regular_tx).set_index("month").reindex(all_monthly["month"],
                                                                               fill_value=0.0).reset_index()
        note_parts = []
        if one_off_ids:
            excluded = transactions[transactions["id"].isin(one_off_ids)]["amount"].sum()
            note_parts.append(f"{len(one_off_ids)} unusual one-off transaction(s) totalling ₹{excluded:,.0f} were "
                              f"excluded from the forecast inputs.")
        if last_month_is_partial(transactions) and len(monthly) > 1:
            note_parts.append(f"{monthly['month'].iloc[-1]} looks incomplete, so it is excluded from the "
                              f"forecast inputs.")
            monthly = monthly.iloc[:-1]
        months = monthly["month"]
        values = monthly["expense"].to_numpy()
        target = str(months.iloc[-1] + 1)

        if len(monthly) < FORECAST_N_LAGS:
            pred = float(values.mean())
            note_parts.insert(0, f"RULE-BASED forecast (average of {len(monthly)} month(s)). "
                                 f"The ML model needs at least {FORECAST_N_LAGS} complete months.")
            hist = pd.DataFrame({"month": months, "actual": all_monthly["expense"].iloc[:len(months)].to_numpy(),
                                 "regular": values, "predicted": np.nan})
            return ForecastResult(target, round(pred, 2), "rule_average", " ".join(note_parts), hist)

        # in-sample-for-user, out-of-sample-for-model backtest for the chart
        lagged = make_lag_features(values, months)
        hist = pd.DataFrame({"month": months, "actual": all_monthly["expense"].iloc[:len(months)].to_numpy(),
                             "regular": values, "predicted": np.nan})
        if not lagged.empty:
            hist.loc[hist["month"].isin(lagged["month"]), "predicted"] = self.predict_rows(lagged).round(2)
        pred = float(self.predict_rows(features_for_next(values, months.iloc[-1]))[0])
        note_parts.insert(0, f"ML forecast (Ridge Regression) using the last {FORECAST_N_LAGS} months.")
        return ForecastResult(target, round(pred, 2), "ml_ridge", " ".join(note_parts), hist)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path = FORECAST_MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path = FORECAST_MODEL_PATH) -> "SpendingForecaster":
        return joblib.load(path)


# ---------------------------------------------------------------- baselines
def baseline_moving_average(X: pd.DataFrame) -> np.ndarray:
    """Average of the previous 3 months."""
    return X["mean_3"].to_numpy()


def baseline_historical_mean(rows: pd.DataFrame) -> np.ndarray:
    """Average of ALL months before the target month (expanding mean)."""
    return rows["hist_mean"].to_numpy()
