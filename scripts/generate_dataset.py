"""Generate every dataset used by FinGuard AI.

Run from the project root:
    python scripts/generate_dataset.py

Outputs
-------
data/generated/transactions_full.csv   36 months x 6 profiles, labelled (synthetic)
data/generated/classifier_train.csv    expense descriptions, merchants disjoint from test
data/generated/classifier_test.csv
data/generated/anomaly_dataset.csv     expenses with split column train/val/test (time-based)
data/generated/monthly_totals.csv      monthly totals (all + regular) per profile, split column
data/demo_normal.csv / demo_anomaly.csv / demo_overspending.csv
data/sample_transactions.csv           a small, deliberately messy upload example
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import DATA_DIR, GENERATED_DIR, RANDOM_SEED  # noqa: E402
from src.data_generator import (generate_demo_scenarios, generate_training_corpus,  # noqa: E402
                                split_classifier_data)

# Time-based cut-offs (the corpus covers 2023-10 .. 2026-09)
ANOMALY_TRAIN_UNTIL = "2025-04-01"   # 18 months to fit Isolation Forest
ANOMALY_VAL_UNTIL = "2025-10-01"     # 6 months to choose the contamination level
FORECAST_TEST_FROM = "2025-10-01"    # last 12 months are the forecast test period


def build_monthly_totals(full: pd.DataFrame) -> pd.DataFrame:
    exp = full[full["type"] == "Expense"].copy()
    exp["month"] = exp["date"].str[:7]
    exp["regular"] = exp["amount"].where(exp["is_anomaly"] == 0, 0.0)
    monthly = exp.groupby(["profile_id", "month"], as_index=False)[["amount", "regular"]].sum()
    # expense = all spending; regular_expense = spending without the one-off anomalies
    monthly = monthly.rename(columns={"amount": "expense", "regular": "regular_expense"})
    monthly["split"] = (monthly["month"] >= FORECAST_TEST_FROM[:7]).map({True: "test", False: "train"})
    return monthly


def build_sample_upload() -> pd.DataFrame:
    """A short realistic upload with problems the validator must handle."""
    rows = [
        ["S001", "2026-07-01", "Monthly Salary", "42000", "Income", "Salary"],
        ["S002", "2026-07-02", "House Rent July", "12000", "Expense", "Rent"],
        ["S003", "03/07/2026", "Swiggy Order", "₹350", "Expense", ""],
        ["S004", "2026-07-04", "Uber Ride", "220", "Debit", ""],
        ["S005", "2026-07-05", "Amazon Shopping", "2,100", "Expense", ""],
        ["S006", "2026-07-07", "Airtel Postpaid bill payment", "799", "Expense", ""],
        ["S007", "2026-07-09", "Apollo Pharmacy medicines", "460", "Expense", ""],
        ["S008", "2026-07-11", "BookMyShow movie tickets", "640", "Expense", ""],
        ["S009", "2026-07-12", "Zomato Order", "410", "Expense", ""],
        ["S009", "2026-07-12", "Zomato Order", "410", "Expense", ""],          # duplicate ID
        ["S010", "2026-07-14", "Coursera course fee", "3200", "Expense", ""],
        ["S011", "not a date", "Ola cab", "180", "Expense", ""],               # invalid date
        ["S012", "2026-07-16", "Blinkit groceries", "abc", "Expense", ""],     # invalid amount
        ["S013", "2026-07-18", "Indian Oil fuel", "1500", "Expense", ""],
        ["S014", "2026-07-20", "Netflix subscription", "649", "Expense", ""],
        ["S015", "2026-07-22", "UPI-DMART-552311@ybl", "1830", "Expense", ""],
        ["S016", "2026-07-25", "Myntra apparel order", "1899", "Expense", ""],
        ["S017", "2026-07-28", "BESCOM Electricity bill", "1240", "Expense", ""],
        ["S018", "2026-08-01", "Monthly Salary", "42000", "Credit", ""],
        ["S019", "2026-08-02", "House Rent August", "12000", "Expense", "Rent"],
        ["S020", "2026-08-05", "Rapido ride", "95", "Expense", ""],
        ["S021", "2026-08-08", "Dominos order", "520", "Expense", ""],
        ["S022", "2026-08-12", "Croma electronics purchase", "16999", "Expense", ""],
        ["S023", "2026-08-15", "Practo consultation", "500", "Expense", ""],
        ["S024", "2026-08-20", "Spotify subscription", "119", "Expense", ""],
    ]
    return pd.DataFrame(rows, columns=["Transaction ID", "Date", "Description", "Amount", "Type", "Category"])


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    full = generate_training_corpus(seed=RANDOM_SEED)
    full.to_csv(GENERATED_DIR / "transactions_full.csv", index=False)

    clf_train, clf_test = split_classifier_data(full, seed=RANDOM_SEED)
    clf_train.to_csv(GENERATED_DIR / "classifier_train.csv", index=False)
    clf_test.to_csv(GENERATED_DIR / "classifier_test.csv", index=False)

    exp = full[full["type"] == "Expense"].copy()
    exp["split"] = "test"
    exp.loc[exp["date"] < ANOMALY_VAL_UNTIL, "split"] = "val"
    exp.loc[exp["date"] < ANOMALY_TRAIN_UNTIL, "split"] = "train"
    exp.to_csv(GENERATED_DIR / "anomaly_dataset.csv", index=False)

    monthly = build_monthly_totals(full)
    monthly.to_csv(GENERATED_DIR / "monthly_totals.csv", index=False)

    for name, df in generate_demo_scenarios().items():
        df.to_csv(DATA_DIR / name, index=False)
    build_sample_upload().to_csv(DATA_DIR / "sample_transactions.csv", index=False)

    summary = {
        "data_type": "SYNTHETIC - generated by src/data_generator.py",
        "seed": RANDOM_SEED,
        "transactions_total": len(full),
        "expenses_total": int((full["type"] == "Expense").sum()),
        "injected_anomalies": int(full["is_anomaly"].sum()),
        "profiles": int(full["profile_id"].nunique()),
        "date_range": [full["date"].min(), full["date"].max()],
        "classifier_train_rows": len(clf_train),
        "classifier_test_rows": len(clf_test),
        "classifier_test_merchants_unseen_in_train": int(
            len(set(clf_test["merchant"]) - set(clf_train["merchant"]))),
        "anomaly_split_counts": exp["split"].value_counts().to_dict(),
        "anomaly_labels_per_split": exp.groupby("split")["is_anomaly"].sum().to_dict(),
        "monthly_points": monthly["split"].value_counts().to_dict(),
        "category_counts": full[full["type"] == "Expense"]["category"].value_counts().to_dict(),
    }
    (GENERATED_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
