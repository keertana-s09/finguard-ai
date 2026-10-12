"""Evaluate every component on held-out data and write reports/.

Run from the project root:
    python scripts/evaluate_models.py

Protocol (no test data is used for training or tuning):
  * Classifier   - train on classifier_train.csv, test on classifier_test.csv
                   (test merchants never appear in training)
  * Isolation F. - fit on 'train' months, choose contamination on 'val' months,
                   report on 'test' months (labels = injected anomalies)
  * Forecaster   - train on target months before 2025-10, test on 2025-10..2026-09
  * Agent        - 500 randomised budget situations checked against independent
                   constraint calculations + the four demo scenarios

Outputs: reports/evaluation_report.md, reports/metrics.json,
         reports/confusion_matrix_ml.csv, reports/confusion_matrix_keyword.csv,
         reports/forecast_test_predictions.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,  # noqa: E402
                             mean_absolute_error, precision_recall_fscore_support)

from src.anomaly_detector import AnomalyDetector, rule_baseline  # noqa: E402
from src.config import (CLASSIFIER_REVIEW_THRESHOLD, DEFAULT_ESSENTIAL_CATEGORIES, EXPENSE_CATEGORIES,  # noqa: E402
                        MAX_CUT_FRACTION, RANDOM_SEED, REPORTS_DIR)
from src.expense_classifier import KeywordBaselineClassifier, train_classifier  # noqa: E402
from src.financial_agent import (ACHIEVABLE, INFEASIBLE, ON_TRACK, AgentSettings,  # noqa: E402
                                 FinancialDecisionAgent, FinancialState)
from src.spending_predictor import (FEATURES, SpendingForecaster, baseline_historical_mean,  # noqa: E402
                                    baseline_moving_average)
from src.training import load_datasets, select_contamination  # noqa: E402

FORECAST_TEST_FROM = "2025-10"


# ====================================================================== classification
def evaluate_classifier(data):
    train, test = data["clf_train"], data["clf_test"]
    model = train_classifier(train)
    pred = model.predict(test["description"])
    y, p = test["category"], pred["predicted_category"]
    kw = KeywordBaselineClassifier().predict(test["description"])

    labels = sorted(EXPENSE_CATEGORIES)
    cm = pd.DataFrame(confusion_matrix(y, p, labels=labels), index=labels, columns=labels)
    cm_kw = pd.DataFrame(confusion_matrix(y, kw, labels=labels), index=labels, columns=labels)
    cm.to_csv(REPORTS_DIR / "confusion_matrix_ml.csv")
    cm_kw.to_csv(REPORTS_DIR / "confusion_matrix_keyword.csv")

    confident = ~pred["needs_review"]
    return {
        "train_rows": len(train), "test_rows": len(test),
        "unseen_test_merchants": int(len(set(test["merchant"]) - set(train["merchant"]))),
        "ml": {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4)},
        "keyword_baseline": {"accuracy": round(accuracy_score(y, kw), 4),
                             "macro_f1": round(f1_score(y, kw, average="macro"), 4)},
        "per_class": {k: {m: round(v, 3) for m, v in d.items()} for k, d in
                      classification_report(y, p, output_dict=True, zero_division=0).items() if k in labels},
        "review_threshold": CLASSIFIER_REVIEW_THRESHOLD,
        "share_flagged_for_review": round(float(pred["needs_review"].mean()), 4),
        "accuracy_confident": round(accuracy_score(y[confident], p[confident]), 4) if confident.any() else None,
        "accuracy_flagged": round(accuracy_score(y[~confident], p[~confident]), 4) if (~confident).any() else None,
        "confusion_matrix": cm.to_dict(),
    }


# ====================================================================== anomaly detection
def _prf(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), "flagged": int(np.sum(p))}


def evaluate_anomaly(data):
    df = data["anomaly"]
    contamination, grid = select_contamination(df)
    fit_rows = df[df["split"] == "train"]
    results = {"contamination_selected_on_validation": contamination, "validation_search": grid,
               "test_rows": int((df["split"] == "test").sum()),
               "test_true_anomalies": int(df.loc[df["split"] == "test", "label"].sum())}
    for name, onehot in [("isolation_forest", False), ("isolation_forest_with_category_onehot", True)]:
        det = AnomalyDetector(contamination=contamination, use_category_onehot=onehot).fit(fit_rows, "profile_id")
        scored = det.score(df, group_col="profile_id")
        test = scored[scored["split"] == "test"]
        results[name] = _prf(test["label"], test["is_anomaly"])
        if name == "isolation_forest":
            results["rule_baseline_3x_typical"] = _prf(test["label"], rule_baseline(test, 3.0))
            examples = test[test["is_anomaly"]].nlargest(5, "anomaly_score")
            results["top_test_examples"] = examples[["date", "description", "category", "amount",
                                                     "typical_amount", "anomaly_score", "label",
                                                     "anomaly_explanation"]].to_dict("records")
    return results


# ====================================================================== forecasting
def evaluate_forecaster(data):
    monthly = data["monthly"]
    rows = SpendingForecaster.build_training_rows(monthly, value_col="regular_expense")
    rows["month_str"] = rows["month"].astype(str)
    train = rows[rows["month_str"] < FORECAST_TEST_FROM]
    test = rows[rows["month_str"] >= FORECAST_TEST_FROM].copy()
    model = SpendingForecaster()
    model.pipeline.fit(train[FEATURES], train["y"] / train["hist_mean"])

    test["ridge"] = model.predict_rows(test)
    test["hist_mean_baseline"] = baseline_historical_mean(test)
    test["moving_avg_3_baseline"] = baseline_moving_average(test)
    test[["profile_id", "month_str", "y", "ridge", "hist_mean_baseline", "moving_avg_3_baseline"]].rename(
        columns={"month_str": "month", "y": "actual"}).round(2).to_csv(
        REPORTS_DIR / "forecast_test_predictions.csv", index=False)

    def score(col):
        err = (test[col] - test["y"]).abs()
        return {"MAE": round(float(err.mean()), 2), "MAPE_percent": round(float((err / test["y"]).mean() * 100), 2)}

    return {"target": "regular monthly spending (one-off anomalies excluded)",
            "train_points": len(train), "test_points": len(test),
            "test_period": f"{test['month_str'].min()} .. {test['month_str'].max()}",
            "mean_actual": round(float(test["y"].mean()), 2),
            "ridge": score("ridge"), "historical_mean_baseline": score("hist_mean_baseline"),
            "moving_average_3_baseline": score("moving_avg_3_baseline")}


# ====================================================================== agent
def _random_state(rng) -> FinancialState:
    income = float(rng.choice([15000, 25000, 30000, 45000, 60000, 90000, 150000]))
    spend_ratio = rng.uniform(0.5, 1.1)
    shares = rng.dirichlet(np.ones(len(EXPENSE_CATEGORIES)))
    predicted = income * spend_ratio
    baseline = {c: round(predicted * s, 2) for c, s in zip(EXPENSE_CATEGORIES, shares)}
    essentials = list(rng.choice(EXPENSE_CATEGORIES, size=rng.integers(2, 6), replace=False))
    return FinancialState(current_month="2026-09", income=income, income_source="test",
                          current_spending=predicted * rng.uniform(0.8, 1.2), current_by_category=baseline,
                          baseline_by_category=baseline, forecast_month="2026-10", predicted_spending=predicted,
                          forecast_method="test", anomalies=[], flagged_txn_ids=[],
                          savings_goal=float(round(income * rng.uniform(0.0, 0.5), -2)), monthly_budget=0.0,
                          essential_categories=essentials)


def evaluate_agent(n_cases: int = 500):
    rng = np.random.default_rng(RANDOM_SEED)
    checks = {"feasibility_correct": 0, "goal_met_when_feasible": 0, "essentials_untouched": 0,
              "caps_respected": 0, "no_cuts_when_on_track": 0, "correct_alert_action": 0, "proposal_saved": 0}
    status_counts = {ON_TRACK: 0, ACHIEVABLE: 0, INFEASIBLE: 0}
    for _ in range(n_cases):
        state = _random_state(rng)
        agent = FinancialDecisionAgent(AgentSettings(savings_goal=state.savings_goal,
                                                     essential_categories=state.essential_categories))
        agent.state = state
        a = agent.evaluate_goals()
        plan = agent.generate_plan()
        status_counts[a["status"]] += 1

        # independent re-computation of the feasibility rule
        limit = state.income - state.savings_goal
        max_cut = sum(v * (0 if c in state.essential_categories else MAX_CUT_FRACTION.get(c, 0))
                      for c, v in state.baseline_by_category.items())
        gap = state.predicted_spending - limit
        expected = ON_TRACK if gap <= 0 else (ACHIEVABLE if gap <= max_cut + 0.01 else INFEASIBLE)
        checks["feasibility_correct"] += expected == a["status"]
        checks["goal_met_when_feasible"] += (a["status"] == INFEASIBLE) or plan["revised_total"] <= limit + 1
        cut = {r["category"]: r["reduction"] for r in plan["reductions"]}
        checks["essentials_untouched"] += all(c not in cut for c in state.essential_categories)
        checks["caps_respected"] += all(
            v <= state.baseline_by_category[c] * MAX_CUT_FRACTION[c] + 0.01 for c, v in cut.items())
        checks["no_cuts_when_on_track"] += a["status"] != ON_TRACK or not cut
        types = {x.get("alert_type") for x in plan["actions"] if x["action"] == "CREATE_ALERT"}
        want = {ON_TRACK: None, ACHIEVABLE: "FORECAST_OVERSPEND", INFEASIBLE: "GOAL_INFEASIBLE"}[a["status"]]
        checks["correct_alert_action"] += (want in types) if want else not ({"FORECAST_OVERSPEND",
                                                                            "GOAL_INFEASIBLE"} & types)
        checks["proposal_saved"] += any(x["action"] == "SAVE_BUDGET_PROPOSAL" for x in plan["actions"])
    return {"cases": n_cases, "status_distribution": status_counts,
            "checks_passed": {k: f"{v}/{n_cases}" for k, v in checks.items()},
            "all_passed": all(v == n_cases for v in checks.values())}


def evaluate_demo_scenarios():
    import tempfile
    from src.pipeline import DEMO_SCENARIOS, FinGuardPipeline
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        pipe = FinGuardPipeline(db_path=Path(tmp) / "eval.db")
        for key, sc in DEMO_SCENARIOS.items():
            a = pipe.load_demo(key)["analysis"]
            created = [r for r in a["action_results"] if r["outcome"] in ("CREATED", "FLAGGED")]
            out[key] = {
                "title": sc["title"], "forecast": a["forecast"]["predicted"], "method": a["forecast"]["method"],
                "income": a["state"]["income"], "limit": a["assessment"]["spending_limit"],
                "status": a["assessment"]["status"], "unusual_high_value": len(a["state"]["anomalies"]),
                "cuts": {r["category"]: r["reduction"] for r in a["plan"]["reductions"]},
                "actions": sorted({f"{r['action']}:{r['target'].split(':')[0]}" for r in created}),
            }
    return out


# ====================================================================== report
def write_report(m):
    c, an, f, ag = m["classification"], m["anomaly"], m["forecast"], m["agent"]
    lines = [
        "# FinGuard AI - Evaluation Report",
        "",
        "> Generated by `python scripts/evaluate_models.py`. All numbers below were produced by running the code "
        "on the **synthetic** dataset (seed 42). They show that the pipeline works as designed; they do **not** "
        "prove accuracy on real bank data.",
        "",
        "## 1. Expense classification (TF-IDF + Logistic Regression)",
        f"Train rows: {c['train_rows']}, test rows: {c['test_rows']}. The {c['unseen_test_merchants']} test "
        "merchants never appear in training (merchant-disjoint split), so this measures generalisation to "
        "unfamiliar merchants.",
        "",
        "| Model | Accuracy | Macro F1 |", "|---|---|---|",
        f"| TF-IDF + Logistic Regression | {c['ml']['accuracy']:.3f} | {c['ml']['macro_f1']:.3f} |",
        f"| Keyword-rule baseline | {c['keyword_baseline']['accuracy']:.3f} | "
        f"{c['keyword_baseline']['macro_f1']:.3f} |",
        "",
        f"Confidence threshold {c['review_threshold']}: {c['share_flagged_for_review']:.1%} of test predictions "
        f"are flagged for user review. Accuracy on confident predictions: {c['accuracy_confident']}; on flagged "
        f"predictions: {c['accuracy_flagged']}.",
        "",
        "Per-class F1 (ML model): " + ", ".join(f"{k} {v['f1-score']:.2f}" for k, v in c["per_class"].items()),
        "",
        "Confusion matrices: `reports/confusion_matrix_ml.csv`, `reports/confusion_matrix_keyword.csv` "
        "(rows = true, columns = predicted).",
        "",
        "## 2. Anomaly detection (Isolation Forest)",
        f"Contamination chosen on validation months: **{an['contamination_selected_on_validation']}** "
        f"(search: {', '.join(str(g['contamination']) + '->' + str(g['val_f1']) for g in an['validation_search'])}).",
        f"Test months: {an['test_rows']} expenses, {an['test_true_anomalies']} injected anomalies.",
        "",
        "| Method | Precision | Recall | F1 | Flagged |", "|---|---|---|---|---|",
    ]
    for name, label in [("isolation_forest", "Isolation Forest (used in app)"),
                        ("isolation_forest_with_category_onehot", "Isolation Forest + one-hot category (ablation)"),
                        ("rule_baseline_3x_typical", "Rule: amount > 3x typical")]:
        r = an[name]
        lines.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['flagged']} |")
    lines += [
        "",
        "Note: the injected anomalies are defined as 5-12x the typical amount, which is exactly what the "
        "ratio rule looks for, so the rule has an unfair advantage in recall on this synthetic data. It also "
        "raises many more alerts (lower precision), because natural spending has long tails. Compare F1 and "
        "the number of flagged transactions, not recall alone.",
        "",
        "## 3. Spending forecast (Ridge Regression)",
        f"Target: {f['target']}. Time-based split - train points: {f['train_points']}, test points: "
        f"{f['test_points']} ({f['test_period']}, 6 profiles). Mean actual monthly spending: "
        f"₹{f['mean_actual']:,.0f}.",
        "",
        "| Method | MAE (₹) | MAPE |", "|---|---|---|",
        f"| Ridge Regression | {f['ridge']['MAE']:,.0f} | {f['ridge']['MAPE_percent']:.1f}% |",
        f"| Historical average baseline | {f['historical_mean_baseline']['MAE']:,.0f} | "
        f"{f['historical_mean_baseline']['MAPE_percent']:.1f}% |",
        f"| 3-month moving average baseline | {f['moving_average_3_baseline']['MAE']:,.0f} | "
        f"{f['moving_average_3_baseline']['MAPE_percent']:.1f}% |",
        "",
        "Interpretation: monthly spending contains a lot of random variation (a few large purchases decide the "
        "total), so even the best model cannot be very precise. Compare Ridge with the baselines above before "
        "claiming an improvement.",
        "",
        "## 4. Goal-based agent",
        f"{ag['cases']} randomised budget situations. Status distribution: {ag['status_distribution']}.",
        "",
        "| Check | Passed |", "|---|---|",
    ] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in ag["checks_passed"].items()] + [
        "",
        "## 5. Demo scenarios (computed live by the pipeline)",
        "",
        "| Scenario | Income | Limit | Forecast | Status | Unusual (high) | Cuts | Persisted actions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in m["demo_scenarios"].values():
        cuts = ", ".join(f"{k} ₹{v:,.0f}" for k, v in s["cuts"].items()) or "-"
        lines.append(f"| {s['title']} | ₹{s['income']:,.0f} | ₹{s['limit']:,.0f} | ₹{s['forecast']:,.0f} | "
                     f"{s['status']} | {s['unusual_high_value']} | {cuts} | {'<br>'.join(s['actions'])} |")
    (REPORTS_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    data = load_datasets()
    metrics = {"data": "SYNTHETIC", "classification": evaluate_classifier(data),
               "anomaly": evaluate_anomaly(data), "forecast": evaluate_forecaster(data),
               "agent": evaluate_agent(), "demo_scenarios": evaluate_demo_scenarios()}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_report(metrics)
    print((REPORTS_DIR / "evaluation_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
