"""Exportable reports built from the database and the latest analysis run."""
from __future__ import annotations

import pandas as pd

from src.database import FinanceDB
from src.utils import format_inr, now_iso


def transactions_csv(db: FinanceDB) -> bytes:
    cols = ["txn_id", "date", "description", "amount", "type", "category", "category_source",
            "category_confidence", "needs_review", "is_anomaly", "anomaly_score", "anomaly_explanation",
            "review_status"]
    return db.get_transactions()[cols].to_csv(index=False).encode("utf-8")


def markdown_report(db: FinanceDB) -> str:
    run = db.latest_run()
    if not run or not run.get("ok"):
        return "# FinGuard AI Report\n\nNo analysis has been run yet.\n"
    st, a, plan, fc = run["state"], run["assessment"], run["plan"], run["forecast"]
    alerts = db.list_alerts()
    recs = db.list_recommendations("ACTIVE")
    lines = [
        "# FinGuard AI - Personal Finance Report",
        f"Generated: {now_iso()}  |  Analysis run: {run['run_id']}",
        "",
        "## Financial summary",
        f"- Latest month: **{st['current_month']}**",
        f"- Monthly income: **{format_inr(st['income'])}** ({st['income_source']})",
        f"- Spending in {st['current_month']}: **{format_inr(st['current_spending'])}**",
        f"- Savings goal: **{format_inr(st['savings_goal'])}**",
        f"- Forecast for {fc['target_month']}: **{format_inr(fc['predicted'] or 0)}** ({fc['method']})",
        f"  - {fc['note']}",
        "",
        "## Agent decision",
        f"**Status: {a['status']}** - {a['reason']}",
        "",
        f"- Spending limit: {format_inr(a['spending_limit'])} ({a['limit_reason']})",
        f"- Maximum possible reduction from flexible categories: {format_inr(a['max_possible_reduction'])}",
        f"- Expected savings with the revised budget: {format_inr(plan['expected_savings'])}",
        "",
        "### Revised budget",
        "| Category | Planned | Reduction | Revised | Essential |",
        "|---|---|---|---|---|",
    ]
    for r in plan["revised_budget"]:
        lines.append(f"| {r['category']} | {format_inr(r['planned'])} | {format_inr(r['reduction'])} | "
                     f"{format_inr(r['revised'])} | {'yes' if r['essential'] else 'no'} |")
    lines += ["", "### Why these adjustments"]
    lines += [f"- **{r['category']}** -{format_inr(r['reduction'])}: {r['reason']}" for r in plan["reductions"]] \
        or ["- No reductions needed."]
    lines += ["", "## Unusual transactions"]
    if st["anomalies"]:
        for an in st["anomalies"]:
            lines.append(f"- {an['date']} {an['description']} ({an['category']}, {format_inr(an['amount'])}, "
                         f"score {an['anomaly_score']:.3f}): {an['explanation']}")
    else:
        lines.append("- None detected.")
    lines += ["", "## Alerts"]
    if alerts.empty:
        lines.append("- No alerts.")
    for r in alerts.itertuples():
        lines.append(f"- [{r.status}] [{r.severity}] {r.title} - {r.message}")
    lines += ["", "## Recommendations"]
    lines += [f"- {r.message}" for r in recs.itertuples()] or ["- None."]
    lines += ["", "---", "_Simulated planning output. FinGuard AI never moves money or changes bank accounts. "
              "Explanations of unusual transactions are based on category statistics; Isolation Forest only "
              "provides the anomaly score._"]
    return "\n".join(lines) + "\n"


def category_summary(db: FinanceDB) -> pd.DataFrame:
    tx = db.get_transactions()
    exp = tx[tx["type"] == "Expense"].copy()
    if exp.empty:
        return pd.DataFrame(columns=["month", "category", "amount"])
    exp["month"] = exp["date"].str[:7]
    return exp.groupby(["month", "category"], as_index=False)["amount"].sum()
