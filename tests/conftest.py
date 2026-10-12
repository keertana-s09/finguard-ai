"""Shared pytest fixtures. Models are loaded once per test session."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import FinGuardPipeline  # noqa: E402
from src.training import load_models  # noqa: E402


@pytest.fixture(scope="session")
def models():
    return load_models()


@pytest.fixture
def pipeline(tmp_path, models):
    return FinGuardPipeline(db_path=tmp_path / "test.db", models=models)


def make_history(months=6, start="2026-01-01", income=50000, extra=None):
    """Small, deterministic transaction history used across tests."""
    rows = []
    month_starts = pd.date_range(start, periods=months, freq="MS")
    n = 0
    for m in month_starts:
        def add(day, desc, amount, typ="Expense", cat=""):
            nonlocal n
            n += 1
            rows.append({"Transaction ID": f"X{n:04d}", "Date": (m + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                         "Description": desc, "Amount": amount, "Type": typ, "Category": cat})
        add(0, "Monthly Salary", income, "Income", "Income")
        add(1, "House Rent", 12000, cat="Rent")
        for d in range(2, 28, 2):
            add(d, "Swiggy Order", 300 + (d % 5) * 20)
        for d in range(3, 28, 4):
            add(d, "Uber Ride", 200 + (d % 3) * 15)
        add(10, "Airtel Postpaid bill payment", 799)
        add(12, "Amazon Shopping", 1800)
        add(20, "Netflix subscription", 649)
    df = pd.DataFrame(rows)
    if extra is not None:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df
