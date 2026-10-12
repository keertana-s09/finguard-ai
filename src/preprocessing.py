"""Data validation and preprocessing for uploaded transaction CSV files.

Responsibilities
----------------
* map flexible column names (e.g. "txn_id", "Transaction ID") to one schema
* parse dates in several common formats (ISO and Indian dd/mm/yyyy)
* parse amounts such as "₹1,250.00"
* normalise transaction type (Income / Expense)
* drop invalid rows with a clear reason, detect duplicates
* aggregate expenses into monthly totals for forecasting
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import EXPENSE_CATEGORIES, INCOME_CATEGORY
from src.utils import stable_hash

STANDARD_COLUMNS = ["txn_id", "date", "description", "amount", "type", "category"]
REQUIRED_COLUMNS = ["date", "description", "amount", "type"]

# lower-cased / stripped header -> standard column
COLUMN_ALIASES = {
    "transaction id": "txn_id", "transaction_id": "txn_id", "txn_id": "txn_id",
    "txn id": "txn_id", "id": "txn_id", "reference": "txn_id",
    "date": "date", "transaction date": "date", "txn date": "date", "value date": "date",
    "description": "description", "narration": "description", "details": "description",
    "particulars": "description", "merchant": "description",
    "amount": "amount", "amount (inr)": "amount", "amount(inr)": "amount",
    "type": "type", "transaction type": "type", "txn type": "type", "dr/cr": "type",
    "category": "category",
}

TYPE_ALIASES = {
    "income": "Income", "credit": "Income", "cr": "Income", "deposit": "Income",
    "expense": "Expense", "debit": "Expense", "dr": "Expense", "withdrawal": "Expense",
}

DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d-%b-%Y", "%d %B %Y"]

_CATEGORY_LOOKUP = {c.lower(): c for c in EXPENSE_CATEGORIES + [INCOME_CATEGORY]}
_CATEGORY_LOOKUP["salary"] = INCOME_CATEGORY


@dataclass
class ValidationResult:
    """Outcome of validating one CSV upload."""
    data: pd.DataFrame
    errors: list = field(default_factory=list)     # fatal problems (nothing imported)
    warnings: list = field(default_factory=list)   # rows dropped / fixed
    n_input: int = 0
    n_invalid: int = 0
    n_duplicates: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.data) > 0


# ---------------------------------------------------------------- text
def clean_description(text: str) -> str:
    """Normalise a bank narration for the text classifier.

    'UPI-SWIGGY-8812331@ybl' -> 'upi swiggy ybl'. Digits are removed because
    reference numbers carry no category information and would cause leakage
    between near-identical training/test rows.
    """
    text = str(text).lower()
    text = re.sub(r"[0-9]+", " ", text)
    text = re.sub(r"[^a-z&' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- parsing helpers
def parse_dates(values: pd.Series) -> pd.Series:
    """Try each known format and keep the first successful parse per row."""
    values = values.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        missing = result.isna()
        if not missing.any():
            break
        result[missing] = pd.to_datetime(values[missing], format=fmt, errors="coerce")
    return result


def parse_amounts(values: pd.Series) -> pd.Series:
    cleaned = (values.astype(str)
               .str.replace("₹", "", regex=False)
               .str.replace("INR", "", regex=False)
               .str.replace("Rs.", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_type(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().map(TYPE_ALIASES)


def normalize_category(values: pd.Series) -> pd.Series:
    """Known categories are title-cased; blanks and unknown labels become NaN
    (so the ML classifier will fill them in)."""
    return values.astype(str).str.strip().str.lower().map(_CATEGORY_LOOKUP)


# ---------------------------------------------------------------- main entry
def validate_transactions(raw: pd.DataFrame) -> ValidationResult:
    """Validate and clean a raw DataFrame read from a CSV upload."""
    result = ValidationResult(data=pd.DataFrame(columns=STANDARD_COLUMNS), n_input=len(raw))
    if raw is None or raw.empty:
        result.errors.append("The file contains no rows.")
        return result

    df = raw.rename(columns=lambda c: COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()))
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.errors.append(
            "Missing required column(s): " + ", ".join(missing)
            + ". Required: Date, Description, Amount, Type (Category and Transaction ID are optional)."
        )
        return result

    for col in ["txn_id", "category"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[STANDARD_COLUMNS].copy()

    df["date"] = parse_dates(df["date"])
    df["amount"] = parse_amounts(df["amount"])
    df["type"] = normalize_type(df["type"])
    df["description"] = df["description"].astype(str).str.strip()
    df.loc[df["description"].isin(["", "nan", "None"]), "description"] = np.nan
    df["category"] = normalize_category(df["category"])
    df["txn_id"] = df["txn_id"].astype(str).str.strip()
    df.loc[df["txn_id"].isin(["", "nan", "None"]), "txn_id"] = np.nan

    # ---- drop invalid rows with explicit reasons
    problems = {
        "unreadable date": df["date"].isna(),
        "missing description": df["description"].isna(),
        "non-numeric amount": df["amount"].isna(),
        "zero or negative amount": df["amount"].notna() & (df["amount"] <= 0),
        "type is not Income/Expense": df["type"].isna(),
    }
    invalid = pd.Series(False, index=df.index)
    for reason, mask in problems.items():
        if mask.any():
            rows = (mask[mask].index + 2).tolist()  # +2: header row + 1-based
            shown = ", ".join(map(str, rows[:10])) + (" ..." if len(rows) > 10 else "")
            result.warnings.append(f"{int(mask.sum())} row(s) dropped - {reason} (CSV line {shown})")
        invalid |= mask
    df = df[~invalid].copy()
    result.n_invalid = int(invalid.sum())

    # Income rows always get the Income category
    df.loc[df["type"] == "Income", "category"] = INCOME_CATEGORY
    df.loc[(df["type"] == "Expense") & (df["category"] == INCOME_CATEGORY), "category"] = np.nan

    # ---- duplicates inside the file
    has_id = df["txn_id"].notna()
    dup_id = has_id & df.duplicated(subset=["txn_id"], keep="first")
    dup_row = ~has_id & df.duplicated(subset=["date", "description", "amount", "type"], keep="first")
    dups = dup_id | dup_row
    if dups.any():
        result.warnings.append(f"{int(dups.sum())} duplicate row(s) inside the file were skipped.")
    df = df[~dups].copy()
    result.n_duplicates = int(dups.sum())

    # ---- generate IDs where absent (deterministic, so re-uploads are detected as duplicates)
    no_id = df["txn_id"].isna()
    if no_id.any():
        df.loc[no_id, "txn_id"] = df[no_id].apply(make_fingerprint, axis=1)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["amount"] = df["amount"].astype(float).round(2)
    result.data = df.sort_values("date").reset_index(drop=True)
    if result.data.empty:
        result.errors.append("No valid rows remain after validation.")
    return result


def make_fingerprint(row) -> str:
    date = pd.Timestamp(row["date"]).strftime("%Y%m%d")
    key = stable_hash([clean_description(row["description"]), float(row["amount"]), row["type"]])
    return f"AUTO-{date}-{key[:8]}"


def load_csv(source) -> ValidationResult:
    """Read a CSV from a path or file-like object and validate it."""
    try:
        raw = pd.read_csv(source, dtype=str, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        return ValidationResult(data=pd.DataFrame(columns=STANDARD_COLUMNS), errors=["The file is empty."])
    except Exception as exc:  # malformed CSV, wrong encoding, etc.
        return ValidationResult(data=pd.DataFrame(columns=STANDARD_COLUMNS),
                                errors=[f"Could not read the file as CSV: {exc}"])
    return validate_transactions(raw)


# ---------------------------------------------------------------- aggregation
def monthly_expense_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per calendar month: month (Period), expense total, income total."""
    if df.empty:
        return pd.DataFrame(columns=["month", "expense", "income"])
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
    out = (d.pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0)
             .rename(columns={"Expense": "expense", "Income": "income"}))
    for col in ["expense", "income"]:
        if col not in out.columns:
            out[col] = 0.0
    # make the month index continuous so a month with no spending counts as 0
    full = pd.period_range(out.index.min(), out.index.max(), freq="M")
    out = out.reindex(full, fill_value=0.0)
    out.index.name = "month"
    return out[["expense", "income"]].reset_index()


def last_month_is_partial(df: pd.DataFrame, min_day: int = 25) -> bool:
    """True if the latest month in the data looks incomplete (last txn before day `min_day`)."""
    if df.empty:
        return False
    dates = pd.to_datetime(df["date"])
    last = dates.max()
    return last.day < min_day and last.days_in_month > min_day
