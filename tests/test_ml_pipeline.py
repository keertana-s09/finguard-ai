"""Tests 4-6: category prediction, unusual transaction detection, forecasting."""
import json

import pandas as pd

from src.expense_classifier import KeywordBaselineClassifier
from src.spending_predictor import make_lag_features


# 4. automatic category prediction
def test_classifier_predicts_obvious_categories(models):
    pred = models["classifier"].predict(["Swiggy Order", "Uber Ride", "Apollo Pharmacy medicines",
                                         "Netflix subscription", "House Rent July", "BESCOM Electricity bill"])
    assert list(pred["predicted_category"]) == ["Food", "Transport", "Healthcare", "Entertainment", "Rent", "Bills"]
    assert pred["confidence"].between(0, 1).all()


def test_low_confidence_is_flagged_for_review(models):
    pred = models["classifier"].predict(["xq zzv"])
    assert pred["confidence"].iloc[0] < 0.6
    assert bool(pred["needs_review"].iloc[0])


def test_blank_categories_are_filled_and_user_labels_kept(pipeline):
    raw = pd.DataFrame({"Date": ["2026-07-01", "2026-07-02", "2026-07-03"],
                        "Description": ["Monthly Salary", "Swiggy Order", "Zomato Order"],
                        "Amount": [40000, 350, 400], "Type": ["Income", "Expense", "Expense"],
                        "Category": ["", "", "Entertainment"]})
    import io
    pipeline.import_csv(io.StringIO(raw.to_csv(index=False)))
    tx = pipeline.classify().set_index("description")
    assert tx.loc["Swiggy Order", "category"] == "Food"
    assert tx.loc["Swiggy Order", "category_source"] == "model"
    assert tx.loc["Zomato Order", "category"] == "Entertainment"   # user label is not overwritten
    assert tx.loc["Zomato Order", "category_source"] == "user"


def test_keyword_baseline_falls_back_to_other():
    assert KeywordBaselineClassifier().predict(["Swiggy Order", "something random"]) == ["Food", "Other"]


# 5. unusual transaction detection
def _expenses(n=40, big=None):
    rows = [{"id": i, "txn_id": f"E{i}", "date": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)).strftime(
        "%Y-%m-%d"), "description": "Swiggy Order", "amount": 300.0 + (i % 7) * 15, "type": "Expense",
        "category": "Food"} for i in range(n)]
    if big:
        rows.append({"id": n, "txn_id": "BIG", "date": "2026-03-01", "description": "Swiggy party order",
                     "amount": big, "type": "Expense", "category": "Food"})
    return pd.DataFrame(rows)


def test_large_transaction_is_flagged_with_explanation(models):
    scored, _ = models["anomaly"].detect(_expenses(big=6000.0))
    big = scored[scored["txn_id"] == "BIG"].iloc[0]
    assert bool(big["is_anomaly"])
    assert big["anomaly_score"] == scored["anomaly_score"].max()
    assert "typical" in big["anomaly_explanation"] and "₹6,000" in big["anomaly_explanation"]
    assert scored[scored["txn_id"] != "BIG"]["is_anomaly"].sum() == 0


def test_income_rows_are_ignored(models):
    df = _expenses()
    df.loc[len(df)] = {"id": 999, "txn_id": "INC", "date": "2026-02-01", "description": "Salary",
                       "amount": 500000.0, "type": "Income", "category": "Income"}
    scored, _ = models["anomaly"].detect(df)
    assert "INC" not in set(scored["txn_id"])


def test_insufficient_history_is_handled(models):
    scored, msg = models["anomaly"].detect(_expenses(n=3))
    assert scored.empty and "skipped" in msg


# 6. spending forecast
def _tx_for_months(values):
    rows = []
    for i, v in enumerate(values):
        month = pd.Period("2026-01", "M") + i
        rows.append({"id": i, "date": f"{month}-28", "amount": float(v), "type": "Expense", "category": "Food"})
    return pd.DataFrame(rows)


def test_ml_forecast_with_enough_history(models):
    res = models["forecaster"].forecast(_tx_for_months([30000, 32000, 31000, 33000, 32000]))
    assert res.method == "ml_ridge"
    assert res.target_month == "2026-06"
    assert 20000 < res.predicted < 45000


def test_rule_based_fallback_with_short_history(models):
    res = models["forecaster"].forecast(_tx_for_months([30000, 34000]))
    assert res.method == "rule_average"
    assert res.predicted == 32000
    assert "RULE-BASED" in res.note


def test_forecast_dict_is_strict_json(models):
    """Months without a backtest prediction must be null, not NaN (NaN is invalid JSON)."""
    out = models["forecaster"].forecast(_tx_for_months([30000, 32000, 31000, 33000])).to_dict()
    assert out["history"][0]["predicted"] is None
    assert out["history"][-1]["predicted"] is not None
    json.dumps(out, allow_nan=False)


def test_one_off_anomalies_are_excluded_from_forecast_inputs(models):
    tx = _tx_for_months([30000, 32000, 31000, 33000])
    base = models["forecaster"].forecast(tx).predicted
    spike = pd.concat([tx, pd.DataFrame([{"id": 99, "date": "2026-04-15", "amount": 40000.0,
                                          "type": "Expense", "category": "Shopping"}])], ignore_index=True)
    assert models["forecaster"].forecast(spike).predicted > base           # spike included
    assert models["forecaster"].forecast(spike, one_off_ids=[99]).predicted == base   # spike excluded


def test_lag_features_never_use_future_months():
    months = pd.period_range("2026-01", periods=6, freq="M")
    a = make_lag_features([10, 20, 30, 40, 50, 60], months)
    b = make_lag_features([10, 20, 30, 40, 50, 999], months)  # change only the LAST month
    cols = ["lag_1", "lag_2", "lag_3", "hist_mean"]
    pd.testing.assert_frame_equal(a[cols], b[cols])            # features unaffected
