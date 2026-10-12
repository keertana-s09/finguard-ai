"""Goal-based intelligent agent for budget decisions.

Agent cycle
-----------
    perceive()        read the environment -> FinancialState (percepts)
    evaluate_goals()  compare the predicted state with the savings goal
    generate_plan()   search over discretionary cuts, build revised budget,
                      decide which actions to request
    execute_actions() hand the actions to the AutomatedResponseEngine

Environment : the user's transactions, ML outputs and settings
Percepts    : income, current spending, forecast, category spending,
              anomaly results, savings goal, essential categories
Goal        : next-month spending <= income - savings goal (and <= budget cap)
Actions     : CREATE_ALERT, FLAG_TRANSACTION, SAVE_BUDGET_PROPOSAL,
              CREATE_RECOMMENDATION

The agent is deterministic and fully rule-based (no LLM). Every decision is
accompanied by a reason string so it can be explained in the UI and the viva.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from src.config import (CUT_PRIORITY, DEFAULT_ESSENTIAL_CATEGORIES, EXPENSE_CATEGORIES, HIGH_SIDE_RATIO,
                        LARGE_TXN_INCOME_SHARE, MAX_CUT_FRACTION)
from src.utils import format_inr

ON_TRACK = "ON_TRACK"
ACHIEVABLE = "ACHIEVABLE_WITH_CUTS"
INFEASIBLE = "INFEASIBLE"


@dataclass
class AgentSettings:
    savings_goal: float
    monthly_income: float = 0.0          # 0 -> use income detected from transactions
    monthly_budget: float = 0.0          # optional spending cap, 0 -> none
    essential_categories: list = field(default_factory=lambda: list(DEFAULT_ESSENTIAL_CATEGORIES))
    max_cut_fraction: dict = field(default_factory=lambda: dict(MAX_CUT_FRACTION))


@dataclass
class FinancialState:
    """The agent's internal model of the world, built from percepts."""
    current_month: str
    income: float
    income_source: str
    current_spending: float
    current_by_category: dict
    baseline_by_category: dict          # typical monthly spend per category (last 3 months, anomalies removed)
    forecast_month: str
    predicted_spending: float
    forecast_method: str
    anomalies: list                     # high-side unusual transactions (dicts)
    flagged_txn_ids: list               # every transaction Isolation Forest flagged (db ids)
    savings_goal: float
    monthly_budget: float
    essential_categories: list


class FinancialDecisionAgent:
    def __init__(self, settings: AgentSettings):
        self.settings = settings
        self.state: FinancialState | None = None
        self.assessment: dict | None = None
        self.plan: dict | None = None

    # ================================================================ 1. PERCEIVE
    def perceive(self, transactions: pd.DataFrame, forecast: dict, scored_expenses: pd.DataFrame) -> FinancialState:
        """transactions: all rows from the DB (with ML categories).
        forecast: ForecastResult.to_dict(). scored_expenses: output of AnomalyDetector."""
        tx = transactions.copy()
        if "review_status" not in tx:
            tx["review_status"] = "NONE"
        tx["month"] = pd.to_datetime(tx["date"]).dt.to_period("M").astype(str)
        current_month = tx["month"].max()
        months = sorted(tx["month"].unique())
        recent = months[-3:]

        # ---- income
        inc = tx[tx["type"] == "Income"].groupby("month")["amount"].sum()
        detected_income = float(inc.reindex(recent, fill_value=0).mean()) if len(inc) else 0.0
        if self.settings.monthly_income > 0:
            income, income_source = float(self.settings.monthly_income), "user setting"
        else:
            income, income_source = detected_income, "detected from last 3 months of income transactions"

        # ---- spending
        exp = tx[tx["type"] == "Expense"]
        current = exp[exp["month"] == current_month]
        current_by_cat = current.groupby("category")["amount"].sum().round(2).to_dict()

        flagged_ids = []
        anomalies = []
        if scored_expenses is not None and not scored_expenses.empty:
            flagged = scored_expenses[scored_expenses["is_anomaly"]]
            flagged_ids = [int(i) for i in flagged["id"]] if "id" in flagged else []
            review = tx.set_index("id")["review_status"] if "id" in tx else pd.Series(dtype=str)
            for r in flagged.itertuples(index=False):
                ratio = r.amount / r.typical_amount if r.typical_amount else 0
                if ratio < HIGH_SIDE_RATIO:
                    continue
                anomalies.append({
                    "id": int(r.id), "txn_id": r.txn_id, "date": r.date, "month": str(r.date)[:7],
                    "description": r.description, "category": r.category, "amount": float(r.amount),
                    "typical_amount": float(r.typical_amount), "ratio": round(float(ratio), 2),
                    "anomaly_score": float(r.anomaly_score), "explanation": r.anomaly_explanation,
                    "review_status": review.get(int(r.id), "NONE"),
                })

        # baseline per category: last 3 months, high-side anomalies removed (one-off spikes
        # should not define what a "normal" month looks like)
        anomaly_ids = {a["id"] for a in anomalies}
        normal_recent = exp[exp["month"].isin(recent) & ~exp["id"].isin(anomaly_ids)]
        baseline = (normal_recent.groupby("category")["amount"].sum() / len(recent)).round(2).to_dict()

        self.state = FinancialState(
            current_month=current_month, income=round(income, 2), income_source=income_source,
            current_spending=round(float(current["amount"].sum()), 2), current_by_category=current_by_cat,
            baseline_by_category=baseline, forecast_month=forecast.get("target_month", "-"),
            predicted_spending=float(forecast.get("predicted") or 0.0), forecast_method=forecast.get("method", "none"),
            anomalies=anomalies, flagged_txn_ids=flagged_ids, savings_goal=float(self.settings.savings_goal),
            monthly_budget=float(self.settings.monthly_budget),
            essential_categories=list(self.settings.essential_categories),
        )
        return self.state

    # ================================================================ 2. EVALUATE GOALS
    def evaluate_goals(self) -> dict:
        s = self.state
        limit = s.income - s.savings_goal
        limit_reason = f"income {format_inr(s.income)} - savings goal {format_inr(s.savings_goal)}"
        if s.monthly_budget > 0 and s.monthly_budget < limit:
            limit = s.monthly_budget
            limit_reason = f"your monthly budget cap of {format_inr(s.monthly_budget)}"

        planned = self._planned_by_category()
        options = []
        for cat, amount in planned.items():
            essential = cat in s.essential_categories
            frac = 0.0 if essential else self.settings.max_cut_fraction.get(cat, 0.0)
            options.append({"category": cat, "planned": round(amount, 2), "essential": essential,
                            "max_cut_fraction": frac, "max_cut": round(amount * frac, 2)})
        max_reduction = sum(o["max_cut"] for o in options)
        essential_total = sum(o["planned"] for o in options if o["essential"])

        gap = s.predicted_spending - limit
        if s.income <= 0:
            status = INFEASIBLE
            reason = "No income was detected or entered, so no savings goal can be met."
        elif gap <= 0:
            status = ON_TRACK
            reason = (f"Predicted spending {format_inr(s.predicted_spending)} is within the limit "
                      f"{format_inr(limit)} ({limit_reason}).")
        elif gap <= max_reduction + 0.01:
            status = ACHIEVABLE
            reason = (f"Predicted spending exceeds the limit by {format_inr(gap)}; flexible categories can be "
                      f"reduced by up to {format_inr(max_reduction)}, so the goal is achievable.")
        else:
            status = INFEASIBLE
            reason = (f"A reduction of {format_inr(gap)} is needed but at most {format_inr(max_reduction)} can be "
                      f"cut from flexible categories without touching essentials "
                      f"({format_inr(essential_total)}).")

        current_anomaly_total = sum(a["amount"] - a["typical_amount"] for a in s.anomalies
                                    if a["month"] == s.current_month)
        self.assessment = {
            "status": status, "reason": reason,
            "spending_limit": round(limit, 2), "limit_reason": limit_reason,
            "predicted_spending": round(s.predicted_spending, 2), "gap": round(gap, 2),
            "max_possible_reduction": round(max_reduction, 2), "essential_total": round(essential_total, 2),
            "projected_savings": round(s.income - s.predicted_spending, 2),
            "max_achievable_savings": round(s.income - (s.predicted_spending - max_reduction), 2),
            "current_over_limit": round(s.current_spending - limit, 2),
            "current_anomaly_excess": round(current_anomaly_total, 2),
            "options": options,
        }
        return self.assessment

    def _planned_by_category(self) -> dict:
        """Split the predicted total across categories using the recent baseline shares."""
        s = self.state
        base = {c: v for c, v in s.baseline_by_category.items() if v > 0}
        total = sum(base.values())
        if total <= 0:
            return {}
        return {c: s.predicted_spending * v / total for c, v in sorted(base.items(), key=lambda kv: -kv[1])}

    # ================================================================ 3. GENERATE PLAN
    def generate_plan(self) -> dict:
        s, a = self.state, self.assessment
        options = {o["category"]: dict(o) for o in a["options"]}
        anomaly_cats = {x["category"] for x in s.anomalies if x["month"] == s.current_month}
        needed = max(0.0, a["gap"]) if a["status"] != INFEASIBLE else a["max_possible_reduction"]

        # order candidates: categories with a recent anomaly first, then by cut priority,
        # then the categories that grew most versus their baseline.
        def growth(cat):
            base = s.baseline_by_category.get(cat, 0)
            return (s.current_by_category.get(cat, 0) - base) / base if base else 0.0

        candidates = [c for c, o in options.items() if o["max_cut"] > 0]
        candidates.sort(key=lambda c: (c not in anomaly_cats, CUT_PRIORITY.get(c, 9), -growth(c)))

        reductions = []
        remaining = needed if s.income > 0 else 0.0
        for cat in candidates:
            if remaining <= 0.005:
                break
            o = options[cat]
            cut = min(remaining, o["max_cut"])
            remaining -= cut
            reasons = [f"{cat} is discretionary (up to {int(o['max_cut_fraction'] * 100)}% can be cut)"]
            if cat in anomaly_cats:
                reasons.append("an unusually high transaction was detected here this month")
            g = growth(cat)
            if g > 0.15:
                reasons.append(f"spending this month is {g:.0%} above your recent average")
            reductions.append({"category": cat, "planned": o["planned"], "reduction": round(cut, 2),
                               "revised": round(o["planned"] - cut, 2), "reason": "; ".join(reasons) + "."})

        cut_map = {r["category"]: r["reduction"] for r in reductions}
        revised_budget = [{"category": c, "planned": o["planned"], "reduction": cut_map.get(c, 0.0),
                           "revised": round(o["planned"] - cut_map.get(c, 0.0), 2), "essential": o["essential"]}
                          for c, o in options.items()]
        revised_total = round(sum(r["revised"] for r in revised_budget), 2)

        self.plan = {
            "status": a["status"], "reason": a["reason"], "target_month": s.forecast_month,
            "reductions": reductions, "revised_budget": revised_budget, "revised_total": revised_total,
            "expected_savings": round(s.income - revised_total, 2),
            "actions": self._decide_actions(reductions, revised_total),
        }
        return self.plan

    # ---------------------------------------------------------------- decision rules -> actions
    def _decide_actions(self, reductions: list, revised_total: float) -> list[dict]:
        s, a = self.state, self.assessment
        actions = []
        month = s.forecast_month

        # Rule 1: forecast overspending that can be fixed -> warning + proposal
        if a["status"] == ACHIEVABLE:
            severity = "HIGH" if a["gap"] > LARGE_TXN_INCOME_SHARE * s.income else "MEDIUM"
            actions.append(_alert("FORECAST_OVERSPEND", f"FORECAST_OVERSPEND:{month}", severity,
                                  f"Projected overspending in {month}",
                                  f"Predicted spending {format_inr(s.predicted_spending)} exceeds your limit of "
                                  f"{format_inr(a['spending_limit'])} by {format_inr(a['gap'])}. A revised budget "
                                  f"that closes the gap has been saved."))

        # Rule 2: goal cannot be met -> high-priority alert + realistic-goal recommendation
        if a["status"] == INFEASIBLE:
            achievable = max(0.0, a["max_achievable_savings"])
            actions.append(_alert("GOAL_INFEASIBLE", f"GOAL_INFEASIBLE:{month}", "HIGH",
                                  f"Savings goal not achievable in {month}",
                                  f"{a['reason']} The most you can realistically save is about "
                                  f"{format_inr(achievable)}."))
            actions.append({"action": "CREATE_RECOMMENDATION", "key": f"REALISTIC_GOAL:{month}",
                            "kind": "REALISTIC_GOAL", "amount": round(achievable, 2),
                            "message": f"Set a savings goal of about {format_inr(achievable)} for {month} "
                                       f"(instead of {format_inr(s.savings_goal)}), or review your essential "
                                       f"expenses / increase income."})

        # Rule 3: already over the limit in the current month (anomalies explain part of it?)
        if a["current_over_limit"] > 0 and s.income > 0:
            excess = a["current_anomaly_excess"]
            msg = (f"You have spent {format_inr(s.current_spending)} in {s.current_month}, "
                   f"{format_inr(a['current_over_limit'])} above the limit of {format_inr(a['spending_limit'])}.")
            if excess > 0:
                msg += f" Unusual transactions account for about {format_inr(excess)} above normal."
            severity = "HIGH" if excess >= 0.5 * a["current_over_limit"] or a["current_over_limit"] > \
                LARGE_TXN_INCOME_SHARE * s.income else "MEDIUM"
            actions.append(_alert("CURRENT_MONTH_OVER_BUDGET", f"CURRENT_MONTH_OVER_BUDGET:{s.current_month}",
                                  severity, f"{s.current_month} spending above limit", msg))

        # Rule 4: unusual transactions -> review flag (+ alert if recent and unreviewed)
        for an in s.anomalies:
            if an["review_status"] == "REVIEWED":
                continue
            actions.append({"action": "FLAG_TRANSACTION", "txn_pk": an["id"],
                            "reason": f"Unusual {an['category']} transaction ({an['ratio']}x typical)"})
            if an["month"] == s.current_month:
                severity = "HIGH" if an["amount"] >= LARGE_TXN_INCOME_SHARE * s.income else "MEDIUM"
                actions.append(_alert("UNUSUAL_TRANSACTION", f"UNUSUAL_TRANSACTION:{an['txn_id']}", severity,
                                      f"Unusual {an['category'].lower()} expense of {format_inr(an['amount'])}",
                                      f"{an['description']} on {an['date']}. {an['explanation']} "
                                      f"Please confirm this transaction.", related_txn=an["id"]))

        # Rule 5: always persist the budget proposal for the forecast month
        if s.predicted_spending > 0:
            actions.append({"action": "SAVE_BUDGET_PROPOSAL", "target_month": month, "feasibility": a["status"],
                            "income": s.income, "savings_goal": s.savings_goal,
                            "spending_limit": a["spending_limit"], "predicted_spending": s.predicted_spending,
                            "revised_total": revised_total,
                            "plan": {"reductions": reductions, "reason": a["reason"],
                                     "forecast_method": s.forecast_method}})

        # Rule 6: savings recommendations
        for r in reductions:
            actions.append({"action": "CREATE_RECOMMENDATION", "key": f"REDUCE:{month}:{r['category']}",
                            "kind": "REDUCE_CATEGORY", "amount": r["reduction"],
                            "message": f"Reduce {r['category']} from {format_inr(r['planned'])} to "
                                       f"{format_inr(r['revised'])} in {month} (save {format_inr(r['reduction'])}): "
                                       f"{r['reason']}"})
        if a["status"] == ON_TRACK and s.income > 0:
            surplus = max(0.0, a["projected_savings"] - s.savings_goal)
            actions.append({"action": "CREATE_RECOMMENDATION", "key": f"ON_TRACK:{month}", "kind": "SAVINGS",
                            "amount": round(a["projected_savings"], 2),
                            "message": f"You are on track to save about {format_inr(a['projected_savings'])} in "
                                       f"{month} (goal {format_inr(s.savings_goal)}). "
                                       + (f"Consider moving the extra {format_inr(surplus)} to an emergency fund "
                                          f"or investment." if surplus > 0 else "")})
        return actions

    # ================================================================ 4. EXECUTE
    def execute_actions(self, engine, run_id: str) -> list[dict]:
        """Delegate the requested actions to the automated response engine."""
        return engine.execute(self.plan["actions"], run_id=run_id)

    def run(self, transactions, forecast, scored_expenses, engine=None, run_id: str = "manual") -> dict:
        self.perceive(transactions, forecast, scored_expenses)
        self.evaluate_goals()
        self.generate_plan()
        results = self.execute_actions(engine, run_id) if engine is not None else []
        return {"state": asdict(self.state), "assessment": self.assessment, "plan": self.plan,
                "action_results": results}


def _alert(alert_type, key, severity, title, message, related_txn=None) -> dict:
    return {"action": "CREATE_ALERT", "alert_type": alert_type, "key": key, "severity": severity,
            "title": title, "message": message, "related_txn": related_txn}


__all__ = ["FinancialDecisionAgent", "AgentSettings", "FinancialState", "ON_TRACK", "ACHIEVABLE", "INFEASIBLE",
           "EXPENSE_CATEGORIES"]
