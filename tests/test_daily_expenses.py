"""Manual entry and current-month budgeting, including retries and month changes."""
from datetime import date, timedelta
from uuid import uuid4

import pandas as pd
import pytest

from src.daily_budget import remaining_monthly_budget


def summary(rows, settings=None, today=date(2026, 10, 27)):
    tx = pd.DataFrame(rows, columns=["date", "amount", "type", "category"])
    return remaining_monthly_budget(tx, settings or {
        "monthly_income": 30000, "savings_goal": 7000,
        "monthly_budget": 0, "essential_categories": ["Rent", "Bills"]}, today)


def test_budget_includes_all_current_expenses_but_not_other_months_or_future_days():
    result = summary([("2026-09-30", 5000, "Expense", "Food"),
                      ("2026-10-01", 10000, "Expense", "Rent"),
                      ("2026-10-27", 500, "Expense", "Food"),
                      ("2026-10-28", 2000, "Expense", "Shopping"),
                      ("2026-11-01", 3000, "Expense", "Food"),
                      ("2026-10-01", 30000, "Income", "Income")])
    assert result["spending_limit"] == 23000
    assert result["spent"] == 10500
    assert result["remaining"] == 12500


def test_budget_cap_and_zero_or_negative_remaining():
    settings = {"monthly_income": 30000, "savings_goal": 7000, "monthly_budget": 20000}
    result = summary([("2026-10-27", 20000, "Expense", "Shopping")], settings)
    assert result["status"] == "AT_LIMIT" and result["remaining"] == 0
    result = summary([("2026-10-27", 20500, "Expense", "Shopping")], settings)
    assert result["status"] == "OVER_BUDGET" and result["remaining"] == -500
    assert "Shopping" in result["suggestions"][0]


def test_month_rollover_preserves_goal_and_resets_spending():
    result = summary([("2026-10-31", 15000, "Expense", "Rent")], today=date(2026, 11, 1))
    assert result["month"] == "2026-11" and result["spent"] == 0
    assert result["remaining"] == 23000


def test_no_income_prompts_for_settings_and_cap_only_is_supported():
    assert summary([], {"monthly_income": 0})["remaining"] is None
    result = summary([], {"monthly_income": 0, "monthly_budget": 12000, "savings_goal": 1000})
    assert result["remaining"] == 12000
    assert "Set monthly income" in result["suggestions"][0]


def test_income_inference_uses_past_income_not_future_and_protects_essentials():
    result = summary([("2026-09-01", 30000, "Income", "Income"),
                      ("2026-11-01", 90000, "Income", "Income"),
                      ("2026-10-01", 24000, "Expense", "Rent")],
                     {"monthly_income": 0, "savings_goal": 7000, "essential_categories": ["Rent"]})
    assert result["income"] == 30000 and result["remaining"] == -1000
    assert not result["suggestions"]  # never recommend reducing protected Rent


def test_manual_entry_reanalyses_and_persists_monthly_recommendation(pipeline):
    pipeline.save_settings(monthly_income=30000, savings_goal=7000)
    result = pipeline.add_expense(date.today().isoformat(), "Lunch at college", 125.50, "Food")
    assert result["ok"] and result["analysis_ok"] and result["inserted"] == 1
    assert result["monthly_budget"]["remaining"] == 22874.50
    saved = pipeline.db.get_transactions().iloc[0]
    assert saved["category"] == "Food" and saved["category_source"] == "user"
    assert pipeline.db.latest_run()["monthly_budget"]["spent"] == 125.50
    active = pipeline.db.list_recommendations("ACTIVE")
    assert len(active[active["kind"] == "MONTHLY_REMAINING"]) == 1
    pipeline.add_expense(date.today().isoformat(), "Bus fare", 25, "Transport")
    assert pipeline.monthly_budget_summary()["remaining"] == 22849.50
    active = pipeline.db.list_recommendations("ACTIVE")
    assert len(active[active["kind"] == "MONTHLY_REMAINING"]) == 1
    assert active[active["kind"] == "MONTHLY_REMAINING"].iloc[0]["amount"] == 22849.50


def test_auto_category_and_intentional_identical_expenses(pipeline):
    today = date.today().isoformat()
    for _ in range(2):
        assert pipeline.add_expense(today, "Swiggy food order", 250)["analysis_ok"]
    tx = pipeline.db.get_transactions()
    assert len(tx) == 2 and tx["txn_id"].nunique() == 2
    assert tx["category"].notna().all() and (tx["category_source"] == "model").all()


def test_retries_do_not_save_twice_and_conflicting_ids_fail(pipeline):
    token = str(uuid4())
    args = (date.today().isoformat(), "Lunch", 100, "Food", token)
    assert pipeline.add_expense(*args)["inserted"] == 1
    assert pipeline.add_expense(*args)["inserted"] == 0
    assert not pipeline.add_expense(args[0], "Different expense", 100, "Food", token)["ok"]
    assert pipeline.db.count_transactions() == 1


@pytest.mark.parametrize("changes", [
    {"amount": 0}, {"amount": -5}, {"amount": float("nan")}, {"amount": float("inf")},
    {"amount": 0.001}, {"description": "   "}, {"description": "x" * 501},
    {"expense_date": "bad-date"}, {"expense_date": (date.today() + timedelta(days=1)).isoformat()},
    {"category": "Crypto"}, {"entry_id": "not-a-uuid"},
])
def test_invalid_entry_does_not_change_database(pipeline, changes):
    args = dict(expense_date=date.today().isoformat(), description="Lunch", amount=100, category="Food")
    args.update(changes)
    assert not pipeline.add_expense(**args)["ok"]
    assert pipeline.db.count_transactions() == 0


def test_saved_expense_survives_analysis_failure_and_retry(pipeline, monkeypatch):
    token = str(uuid4())
    def fail():
        raise RuntimeError("analysis unavailable")
    monkeypatch.setattr(pipeline, "run_analysis", fail)
    args = (date.today().isoformat(), "Lunch", 100, "Food", token)
    result = pipeline.add_expense(*args)
    assert result["ok"] and not result["analysis_ok"]
    assert pipeline.add_expense(*args)["inserted"] == 0
    assert pipeline.db.count_transactions() == 1


def test_streamlit_manual_entry_refreshes_budget(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from src import config
    from src.database import FinanceDB

    db_path = tmp_path / "daily-app.db"
    db = FinanceDB(db_path)
    db.save_settings({"monthly_income": 30000, "savings_goal": 7000})
    monkeypatch.setattr(config, "DEFAULT_DB_PATH", db_path)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=300).run()
        assert not app.exception
        next(x for x in app.text_input if x.label == "Expense description").set_value("Lunch at college")
        next(x for x in app.number_input if x.label == "Expense amount (₹)").set_value(100.0)
        next(x for x in app.selectbox if x.label == "Expense category").select("Food")
        next(x for x in app.button if x.label == "Save expense").click().run()
        assert not app.exception
        assert db.count_transactions() == 1
        assert db.latest_run()["monthly_budget"]["remaining"] == 22900
        assert any(x.label == "Remaining budget" and "22,900" in x.value for x in app.metric)
    finally:
        st.cache_resource.clear()
