"""Central configuration: paths, categories and tunable constants.

Every other module imports from here so that a value is defined in one place.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
# Override with the FINGUARD_DB environment variable (the tests use a temporary file).
DEFAULT_DB_PATH = Path(os.environ.get("FINGUARD_DB", PROJECT_ROOT / "finguard.db"))

CLASSIFIER_PATH = MODELS_DIR / "expense_classifier.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"
FORECAST_MODEL_PATH = MODELS_DIR / "spending_forecaster.joblib"

RANDOM_SEED = 42

# ---------------------------------------------------------------- categories
EXPENSE_CATEGORIES = [
    "Food", "Transport", "Shopping", "Bills", "Entertainment",
    "Education", "Healthcare", "Rent", "Other",
]
INCOME_CATEGORY = "Income"

# Categories the agent must NOT cut unless the user says otherwise.
DEFAULT_ESSENTIAL_CATEGORIES = ["Rent", "Bills", "Education", "Healthcare"]

# Maximum fraction of a discretionary category the agent may cut in one month.
# These are planning assumptions (documented in docs/ml_explanation.md),
# not learned values.
MAX_CUT_FRACTION = {
    "Entertainment": 0.50,
    "Shopping": 0.50,
    "Other": 0.40,
    "Food": 0.25,
    "Transport": 0.20,
    "Bills": 0.10,
    "Education": 0.10,
    "Healthcare": 0.0,
    "Rent": 0.0,
}

# Lower number = cut first. Used to order reductions.
CUT_PRIORITY = {
    "Entertainment": 1,
    "Shopping": 1,
    "Other": 2,
    "Food": 3,
    "Transport": 4,
    "Bills": 5,
    "Education": 6,
    "Healthcare": 7,
    "Rent": 8,
}

# ---------------------------------------------------------------- ML settings
CLASSIFIER_REVIEW_THRESHOLD = 0.60   # below this confidence -> needs user review
FORECAST_N_LAGS = 3                  # months of history the Ridge model needs
MIN_EXPENSES_FOR_ANOMALY = 10        # fewer than this -> anomaly detection skipped
MIN_CATEGORY_HISTORY = 5             # prior txns in a category to trust its median
HIGH_SIDE_RATIO = 2.0                # flagged txn this many x typical = "unusually high"
LARGE_TXN_INCOME_SHARE = 0.10        # anomaly > 10% of income -> high severity

# Alert types the agent "owns": they are auto-resolved when the condition disappears.
AGENT_ALERT_TYPES = [
    "FORECAST_OVERSPEND",
    "GOAL_INFEASIBLE",
    "CURRENT_MONTH_OVER_BUDGET",
    "UNUSUAL_TRANSACTION",
]
