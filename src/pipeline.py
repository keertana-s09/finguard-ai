"""FinGuard pipeline: connects every module in the order of the architecture.

    CSV -> preprocessing -> SQLite -> classifier -> anomaly detector -> forecaster
        -> FinancialDecisionAgent -> AutomatedResponseEngine -> SQLite -> dashboard

The Streamlit app, the demo scenarios and the integration tests all call this
class, so the demo shows exactly what the real code does.
"""
from __future__ import annotations

import uuid

import pandas as pd

from src.anomaly_detector import high_side_anomalies
from src.automated_response import AutomatedResponseEngine
from src.config import DATA_DIR, DEFAULT_DB_PATH, DEFAULT_ESSENTIAL_CATEGORIES
from src.database import FinanceDB
from src.financial_agent import AgentSettings, FinancialDecisionAgent
from src.preprocessing import load_csv
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
        decision = agent.run(tx, forecast, scored, engine=self.engine, run_id=run_id)

        result = {
            "run_id": run_id, "ok": True, "created_at": now_iso(),
            "anomaly_message": anomaly_msg,
            "n_flagged": int(scored["is_anomaly"].sum()) if not scored.empty else 0,
            "n_low_confidence": int(tx["needs_review"].fillna(0).astype(int).sum()),
            "forecast": forecast,
            "settings": s,
            **decision,
        }
        self.db.save_run(run_id, result)
        return result
