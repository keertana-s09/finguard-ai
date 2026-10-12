"""Automated Response Engine.

Receives action requests from the FinancialDecisionAgent and performs real,
persistent software actions in SQLite:

    CREATE_ALERT           insert / update / re-activate an alert (deduplicated by key)
    FLAG_TRANSACTION       mark a transaction PENDING_REVIEW
    SAVE_BUDGET_PROPOSAL   store a revised budget (older proposal -> SUPERSEDED)
    CREATE_RECOMMENDATION  store a savings recommendation (deduplicated by key)

After each run it also SYNCHRONISES state: agent alerts / recommendations /
review flags whose condition no longer holds are automatically resolved.

Every outcome is written to the action_log table. No real money is moved and
no bank account is touched - all actions are simulated planning artefacts.
"""
from __future__ import annotations

from src.config import AGENT_ALERT_TYPES
from src.database import FinanceDB
from src.utils import stable_hash


class AutomatedResponseEngine:
    def __init__(self, db: FinanceDB):
        self.db = db

    def execute(self, actions: list[dict], run_id: str = "manual") -> list[dict]:
        results = []
        alert_keys, rec_keys, flagged = set(), set(), set()
        handlers = {
            "CREATE_ALERT": self._create_alert,
            "FLAG_TRANSACTION": self._flag_transaction,
            "SAVE_BUDGET_PROPOSAL": self._save_proposal,
            "CREATE_RECOMMENDATION": self._create_recommendation,
        }
        for action in actions:
            kind = action.get("action")
            handler = handlers.get(kind)
            if handler is None:
                results.append(self._log(run_id, kind or "UNKNOWN", "-", "SKIPPED", "Unknown action type"))
                continue
            results.append(handler(action, run_id))
            if kind == "CREATE_ALERT":
                alert_keys.add(action["key"])
            elif kind == "CREATE_RECOMMENDATION":
                rec_keys.add(action["key"])
            elif kind == "FLAG_TRANSACTION":
                flagged.add(int(action["txn_pk"]))

        results.extend(self._sync(run_id, alert_keys, rec_keys, flagged))
        return results

    # ---------------------------------------------------------------- handlers
    def _create_alert(self, a: dict, run_id: str) -> dict:
        existing = self.db.get_alert(a["key"])
        if existing is None:
            self.db.insert_alert(a)
            return self._log(run_id, "CREATE_ALERT", a["key"], "CREATED", a["title"])
        if existing["status"] == "RESOLVED":
            self.db.update_alert(a["key"], status="ACTIVE", severity=a["severity"], title=a["title"],
                                 message=a["message"], resolved_reason=None)
            return self._log(run_id, "CREATE_ALERT", a["key"], "UPDATED", "Condition re-appeared; alert re-activated")
        if existing["message"] != a["message"] or existing["severity"] != a["severity"]:
            # keep ACKNOWLEDGED status - the user has already seen it
            self.db.update_alert(a["key"], severity=a["severity"], title=a["title"], message=a["message"])
            return self._log(run_id, "CREATE_ALERT", a["key"], "UPDATED", "Alert details refreshed")
        return self._log(run_id, "CREATE_ALERT", a["key"], "UNCHANGED", "Duplicate alert prevented")

    def _flag_transaction(self, a: dict, run_id: str) -> dict:
        pk = int(a["txn_pk"])
        status = self.db.get_review_status(pk)
        if status is None:
            return self._log(run_id, "FLAG_TRANSACTION", str(pk), "SKIPPED", "Transaction not found")
        if status == "REVIEWED":
            return self._log(run_id, "FLAG_TRANSACTION", str(pk), "SKIPPED", "Already reviewed by user")
        if status == "PENDING_REVIEW":
            return self._log(run_id, "FLAG_TRANSACTION", str(pk), "UNCHANGED", "Already pending review")
        self.db.set_review_status(pk, "PENDING_REVIEW", a.get("reason", ""))
        return self._log(run_id, "FLAG_TRANSACTION", str(pk), "FLAGGED", a.get("reason", ""))

    def _save_proposal(self, a: dict, run_id: str) -> dict:
        p = dict(a)
        p["plan_hash"] = stable_hash([p["feasibility"], round(p["income"]), round(p["savings_goal"]),
                                      round(p["spending_limit"]), round(p["revised_total"]),
                                      [(r["category"], round(r["reduction"])) for r in p["plan"]["reductions"]]])
        latest = self.db.latest_proposal(p["target_month"])
        if latest and latest["plan_hash"] == p["plan_hash"]:
            return self._log(run_id, "SAVE_BUDGET_PROPOSAL", p["target_month"], "UNCHANGED",
                             f"Identical proposal #{latest['id']} already saved")
        new_id = self.db.insert_proposal(p)
        return self._log(run_id, "SAVE_BUDGET_PROPOSAL", p["target_month"], "CREATED",
                         f"Proposal #{new_id} ({p['feasibility']}) saved")

    def _create_recommendation(self, a: dict, run_id: str) -> dict:
        outcome = self.db.upsert_recommendation(a["key"], a["kind"], a["message"], a.get("amount"))
        return self._log(run_id, "CREATE_RECOMMENDATION", a["key"], outcome, a["message"][:80])

    # ---------------------------------------------------------------- synchronisation
    def _sync(self, run_id: str, alert_keys: set, rec_keys: set, flagged: set) -> list[dict]:
        results = []
        for alert in self.db.open_alerts_of_types(AGENT_ALERT_TYPES):
            if alert["alert_key"] not in alert_keys:
                self.db.update_alert(alert["alert_key"], status="RESOLVED",
                                     resolved_reason="Condition no longer present in the latest analysis")
                results.append(self._log(run_id, "RESOLVE_ALERT", alert["alert_key"], "RESOLVED",
                                         "Condition no longer present"))
        n = self.db.resolve_recommendations_except(sorted(rec_keys))
        if n:
            results.append(self._log(run_id, "RESOLVE_RECOMMENDATIONS", "-", "RESOLVED",
                                     f"{n} outdated recommendation(s) resolved"))
        pending = self.db.query_df("SELECT id FROM transactions WHERE review_status='PENDING_REVIEW'")
        for pk in pending["id"]:
            if int(pk) not in flagged:
                self.db.set_review_status(int(pk), "NONE", "")
                results.append(self._log(run_id, "CLEAR_REVIEW_FLAG", str(pk), "RESOLVED",
                                         "No longer flagged as unusual"))
        return results

    def _log(self, run_id, action, target, outcome, detail) -> dict:
        self.db.log_action(run_id, action, target, outcome, detail)
        return {"action": action, "target": target, "outcome": outcome, "detail": detail}
