"""Current calendar month's remaining budget, separate from next-month forecasts."""
from datetime import date

import pandas as pd


def format_daily_amount(value):
    return f"₹{value:,.2f}"


def remaining_monthly_budget(transactions, settings, as_of=None):
    today = as_of or date.today()
    month = today.strftime("%Y-%m")
    tx = transactions.copy()
    if not tx.empty:
        tx = tx[pd.to_datetime(tx["date"]).dt.date <= today]
    current = tx[tx["date"].str.startswith(month)] if not tx.empty else tx
    expenses = current[current["type"] == "Expense"] if not current.empty else current
    spent = round(float(expenses["amount"].sum()), 2) if not expenses.empty else 0.0

    income = float(settings.get("monthly_income", 0))
    source = "monthly income setting"
    if income <= 0:
        inc = tx[tx["type"] == "Income"] if not tx.empty else tx
        # A new month without a salary entry must not erase the known income.
        if not inc.empty:
            totals = inc.groupby(inc["date"].str[:7])["amount"].sum().sort_index()
            income = float(totals.tail(3).mean())
        source = "average of the latest 3 recorded income months"
    goal = float(settings.get("savings_goal", 0))
    cap = float(settings.get("monthly_budget", 0))
    limit = income - goal if income > 0 else (cap if cap > 0 else None)
    if limit is not None and cap > 0:
        limit = min(limit, cap)
    remaining = round(limit - spent, 2) if limit is not None else None
    suggestions = []
    if limit is None:
        status = "NEEDS_BUDGET"
        message = "Set your monthly income and savings goal, or a monthly spending cap, in AI Financial Advisor."
    elif remaining < 0:
        status = "OVER_BUDGET"
        message = (f"Recorded expenses exceed this month's spending limit by {format_daily_amount(-remaining)}. "
                   "Reduce further non-essential purchases and review your budget.")
    elif remaining == 0:
        status = "AT_LIMIT"
        message = "You have used this month's spending budget. Avoid further non-essential purchases."
    else:
        status = "WITHIN_BUDGET"
        message = (f"You have {format_daily_amount(remaining)} left for the rest of {month}. "
                   "Keep upcoming expenses within this amount and account for bills still due.")
    if limit is not None and not expenses.empty and (remaining <= 0 or spent >= 0.8 * max(limit, 0)):
        essential = settings.get("essential_categories", [])
        flexible = expenses[~expenses["category"].isin(essential)]
        amounts = flexible.groupby("category")["amount"].sum().sort_values(ascending=False)
        if not amounts.empty:
            category = str(amounts.index[0])
            suggestions.append(f"Review {category} first: {format_daily_amount(amounts.iloc[0])} recorded this month. "
                               "Keep essential expenses protected.")
    if income <= 0 and cap > 0:
        suggestions.append("This uses your spending cap. Set monthly income to assess the savings goal too.")
    return {"month": month, "as_of": today.isoformat(), "income": round(income, 2),
            "income_source": source, "savings_goal": goal,
            "spending_limit": round(limit, 2) if limit is not None else None,
            "spent": spent, "remaining": remaining, "status": status,
            "recommendation": message, "suggestions": suggestions}
