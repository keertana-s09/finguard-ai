"""FinGuard pipeline: connects every module in the order of the architecture.

    CSV -> preprocessing -> SQLite -> classifier -> anomaly detector -> forecaster
        -> FinancialDecisionAgent -> AutomatedResponseEngine -> SQLite -> dashboard

The Streamlit app, the demo scenarios and the integration tests all call this
class, so the demo shows exactly what the real code does.
"""
from __future__ import annotations

import uuid
import math
from datetime import date

import pandas as pd

from src.anomaly_detector import high_side_anomalies
from src.automated_response import AutomatedResponseEngine
from src.config import DATA_DIR, DEFAULT_DB_PATH, DEFAULT_ESSENTIAL_CATEGORIES, EXPENSE_CATEGORIES
from src.database import FinanceDB
from src.daily_budget import remaining_monthly_budget
from src.financial_agent import AgentSettings, FinancialDecisionAgent
from src.preprocessing import load_csv, validate_transactions
from src.training import load_models
from src.utils import now_iso

DEFAULT_SETTINGS = {
    "monthly_income": 0.0,      # 0 = detect from data
    "savings_goal": 10000.0,
    "monthly_budget": 0.0,      # 0 = no extra cap
    "essential_categories": list(DEFAULT_ESSENTIAL_CATEGORIES),
}

# Demo scenarios: an input file + user settings. Outputs are NOT stored here -
# they are computed by the pipeline every time.
DEMO_SCENARIOS = {
    "normal": {"file": "demo_normal.csv", "title": "1. Normal spending behaviour",
               "settings": {"monthly_income": 0.0, "savings_goal": 10000.0, "monthly_budget": 0.0},
               "story": "A salaried user (~₹55k/month) with steady spending and a ₹10,000 savings goal."},
    "anomaly": {"file": "demo_anomaly.csv", "title": "2. Unusual high-value transactions",
                "settings": {"monthly_income": 0.0, "savings_goal": 10000.0, "monthly_budget": 0.0},
                "story": "Same user, but September contains a ₹6,800 food order and a ₹38,500 electronics "
                         "purchase."},
    "overspending": {"file": "demo_overspending.csv", "title": "3. Overspending with an achievable goal",
                     "settings": {"monthly_income": 30000.0, "savings_goal": 7000.0, "monthly_budget": 0.0},
                     "story": "₹30,000 income, shopping and entertainment rising every month, ₹7,000 goal."},
    "infeasible": {"file": "demo_overspending.csv", "title": "4. Same user, unrealistic goal",
                   "settings": {"monthly_income": 30000.0, "savings_goal": 15000.0, "monthly_budget": 0.0},
                   "story": "Same data as scenario 3 but the user wants to save ₹15,000 - the agent must "
                            "report that this is infeasible."},
}


class FinGuardPipeline:
    def __init__(self, db_path=DEFAULT_DB_PATH, models: dict | None = None):
        self.db = FinanceDB(db_path)
        self.models = models or load_models()
        self.engine = AutomatedResponseEngine(self.db)

    # ---------------------------------------------------------------- settings
    def get_settings(self) -> dict:
        settings = dict(DEFAULT_SETTINGS)
        settings.update(self.db.get_settings())
        return settings

    def save_settings(self, **values):
        self.db.save_settings(values)

    # ---------------------------------------------------------------- ingestion
    def monthly_budget_summary(self, as_of=None) -> dict:
        return remaining_monthly_budget(self.db.get_transactions(), self.get_settings(), as_of)

    def add_expense(self, expense_date, description, amount, category=None, entry_id=None) -> dict:
        """Save one manual expense, then refresh analysis. Retry IDs prevent double saves."""
        try:
            day = date.fromisoformat(str(expense_date))
            if day > date.today():
                raise ValueError("Expense date cannot be in the future.")
            value = float(amount)
            if not math.isfinite(value) or value <= 0 or round(value, 2) <= 0:
                raise ValueError("Amount must be a positive, finite value of at least ₹0.01.")
            description = str(description or "").strip()
            if not description or len(description) > 500:
                raise ValueError("Enter a description between 1 and 500 characters.")
            if category and category not in EXPENSE_CATEGORIES:
                raise ValueError("Choose a valid expense category.")
            token = str(uuid.UUID(str(entry_id))) if entry_id else str(uuid.uuid4())
        except (ValueError, TypeError) as exc:
            return {"ok": False, "errors": [str(exc)], "inserted": 0}
        txn_id = f"MANUAL-{token}"
        rows = pd.DataFrame([{"txn_id": txn_id, "date": day.isoformat(),
                              "description": description, "amount": value,
                              "type": "Expense", "category": category}])
        validated = validate_transactions(rows)
        if not validated.ok:
            return {"ok": False, "errors": validated.errors + validated.warnings, "inserted": 0}
        existing = self.db.get_transactions()
        same_id = existing[existing["txn_id"] == txn_id]
        if not same_id.empty:
            previous = same_id.iloc[0]
            if (previous["date"] != day.isoformat() or previous["description"] != description
                    or float(previous["amount"]) != round(value, 2)):
                return {"ok": False, "errors": ["This entry ID already belongs to a different expense."],
                        "inserted": 0}
        inserted, _ = self.db.insert_transactions(validated.data)
        result = {"ok": True, "inserted": inserted, "errors": []}
        try:
            analysis = self.run_analysis()
            result["analysis_ok"] = bool(analysis.get("ok"))
        except Exception:
            # The expense is already saved. Do not invite the user to enter it again.
            result["analysis_ok"] = False
        result["monthly_budget"] = self.monthly_budget_summary()
        return result

    def import_csv(self, source) -> dict:
        result = load_csv(source)
        if not result.ok:
            return {"ok": False, "errors": result.errors, "warnings": result.warnings,
                    "inserted": 0, "duplicates_in_db": 0, "n_input": result.n_input}
        inserted, dup_db = self.db.insert_transactions(result.data)
        warnings = list(result.warnings)
        if dup_db:
            warnings.append(f"{dup_db} transaction(s) already existed in the database and were skipped.")
        return {"ok": True, "errors": [], "warnings": warnings, "inserted": inserted,
                "duplicates_in_db": dup_db, "duplicates_in_file": result.n_duplicates,
                "invalid_rows": result.n_invalid, "n_input": result.n_input}

    def load_demo(self, scenario: str) -> dict:
        sc = DEMO_SCENARIOS[scenario]
        self.db.reset_all()
        settings = dict(DEFAULT_SETTINGS)
        settings.update(sc["settings"])
        self.db.save_settings(settings)
        imported = self.import_csv(DATA_DIR / sc["file"])
        analysis = self.run_analysis()
        return {"import": imported, "analysis": analysis}

    # ---------------------------------------------------------------- analysis
    def classify(self) -> pd.DataFrame:
        """Predict categories for every expense; fill in only rows without a user label."""
        tx = self.db.get_transactions()
        exp = tx[tx["type"] == "Expense"].copy()
        if exp.empty:
            return tx
        pred = self.models["classifier"].predict(exp["description"]).set_index(exp.index)
        exp["predicted_category"] = pred["predicted_category"]
        exp["category_confidence"] = pred["confidence"]
        keep_user = exp["category_source"].isin(["user", "corrected"]) & exp["category"].notna()
        exp.loc[~keep_user, "category"] = exp.loc[~keep_user, "predicted_category"]
        exp.loc[~keep_user, "category_source"] = "model"
        exp["needs_review"] = (~keep_user) & pred["needs_review"]
        self.db.update_classification(exp[["id", "category", "category_source", "predicted_category",
                                           "category_confidence", "needs_review"]])
        return self.db.get_transactions()

    def run_analysis(self) -> dict:
        """Run the full data -> decision -> action workflow and persist everything."""
        run_id = now_iso().replace(":", "") + "-" + uuid.uuid4().hex[:6]
        if self.db.count_transactions() == 0:
            return {"run_id": run_id, "ok": False, "message": "No transactions loaded yet."}

        # Step 4: ML classification
        tx = self.classify()

        # Step 5: anomaly detection (Isolation Forest)
        scored, anomaly_msg = self.models["anomaly"].detect(tx)
        if not scored.empty:
            self.db.update_anomalies(scored[["id", "anomaly_score", "is_anomaly", "typical_amount",
                                             "anomaly_explanation"]])
        tx = self.db.get_transactions()

        # Step 6: forecast regular spending (anomaly detector output feeds the forecaster)
        one_offs = high_side_anomalies(scored)
        one_off_ids = one_offs["id"].tolist() if one_offs is not None and not one_offs.empty else []
        forecast = self.models["forecaster"].forecast(tx, one_off_ids=one_off_ids).to_dict()

        # Steps 7-10: agent perceives, decides and requests actions; engine executes them
        s = self.get_settings()
        agent = FinancialDecisionAgent(AgentSettings(
            savings_goal=float(s["savings_goal"]), monthly_income=float(s["monthly_income"]),
            monthly_budget=float(s["monthly_budget"]), essential_categories=list(s["essential_categories"])))
        decision = agent.run(tx, forecast, scored)
        daily = remaining_monthly_budget(tx, s)
        if daily["spending_limit"] is not None:
            decision["plan"]["actions"].append({
                "action": "CREATE_RECOMMENDATION", "key": f"MONTHLY_REMAINING:{daily['month']}",
                "kind": "MONTHLY_REMAINING", "amount": daily["remaining"],
                "message": daily["recommendation"] + " " + " ".join(daily["suggestions"]),
            })
        decision["action_results"] = self.engine.execute(decision["plan"]["actions"], run_id=run_id)

        result = {
            "run_id": run_id, "ok": True, "created_at": now_iso(),
            "anomaly_message": anomaly_msg,
            "n_flagged": int(scored["is_anomaly"].sum()) if not scored.empty else 0,
            "n_low_confidence": int(tx["needs_review"].fillna(0).astype(int).sum()),
            "forecast": forecast,
            "settings": s,
            "monthly_budget": daily,
            **decision,
        }
        self.db.save_run(run_id, result)
        return result
