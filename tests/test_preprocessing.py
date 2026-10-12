"""Tests 1-3: CSV validation, missing columns, duplicate handling."""
import io

import pandas as pd

from src.preprocessing import clean_description, load_csv, monthly_expense_totals, validate_transactions
from tests.conftest import make_history


def to_csv(df) -> io.StringIO:
    return io.StringIO(df.to_csv(index=False))


# 1. valid CSV upload
def test_valid_csv_is_accepted():
    res = load_csv(to_csv(make_history(months=2)))
    assert res.ok
    assert res.n_invalid == 0
    assert set(res.data["type"]) == {"Income", "Expense"}
    assert (res.data.loc[res.data["type"] == "Income", "category"] == "Income").all()


def test_mixed_formats_are_parsed():
    raw = pd.DataFrame({"Date": ["2026-07-01", "03/07/2026", "05 Jul 2026"],
                        "Description": ["Salary", "Swiggy Order", "Uber Ride"],
                        "Amount": ["42,000", "₹350", "220.50"],
                        "Type": ["Credit", "Expense", "debit"]})
    res = validate_transactions(raw)
    assert res.ok and len(res.data) == 3
    assert list(res.data["date"]) == ["2026-07-01", "2026-07-03", "2026-07-05"]
    assert list(res.data["amount"]) == [42000.0, 350.0, 220.5]
    assert list(res.data["type"]) == ["Income", "Expense", "Expense"]


def test_invalid_rows_are_dropped_with_reason():
    raw = pd.DataFrame({"Date": ["2026-07-01", "not a date", "2026-07-03", "2026-07-04"],
                        "Description": ["Swiggy", "Ola", "Blinkit", "Zomato"],
                        "Amount": ["100", "50", "abc", "-20"],
                        "Type": ["Expense", "Expense", "Expense", "Expense"]})
    res = validate_transactions(raw)
    assert len(res.data) == 1 and res.n_invalid == 3
    joined = " ".join(res.warnings)
    assert "unreadable date" in joined and "non-numeric amount" in joined and "negative" in joined


# 2. missing columns
def test_missing_required_columns_is_rejected():
    raw = pd.DataFrame({"Date": ["2026-07-01"], "Description": ["Swiggy"]})
    res = validate_transactions(raw)
    assert not res.ok
    assert "Amount" in res.errors[0] or "amount" in res.errors[0]


def test_empty_and_garbage_files():
    assert not load_csv(io.StringIO("")).ok
    assert not load_csv(io.StringIO("just some text\n")).ok


# 3. duplicate handling
def test_duplicates_inside_file_are_removed():
    df = make_history(months=1)
    df = pd.concat([df, df.iloc[[3, 4]]], ignore_index=True)
    res = load_csv(to_csv(df))
    assert res.n_duplicates == 2
    assert len(res.data) == len(df) - 2


def test_duplicates_against_database_are_skipped(pipeline):
    df = make_history(months=1)
    first = pipeline.import_csv(to_csv(df))
    second = pipeline.import_csv(to_csv(df))
    assert first["inserted"] == len(df)
    assert second["inserted"] == 0 and second["duplicates_in_db"] == len(df)


def test_rows_without_ids_get_stable_fingerprints(pipeline):
    df = make_history(months=1).drop(columns=["Transaction ID"])
    assert pipeline.import_csv(to_csv(df))["inserted"] == len(df)
    assert pipeline.import_csv(to_csv(df))["inserted"] == 0


def test_clean_description_removes_reference_numbers():
    assert clean_description("UPI-SWIGGY-8812331@ybl") == "upi swiggy ybl"


def test_monthly_totals_fill_missing_months():
    df = pd.DataFrame({"date": ["2026-01-05", "2026-03-05"], "amount": [100.0, 200.0],
                       "type": ["Expense", "Expense"]})
    m = monthly_expense_totals(df)
    assert list(m["expense"]) == [100.0, 0.0, 200.0]
