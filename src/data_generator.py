"""Synthetic transaction generator (Indian context, amounts in INR).

The data is SYNTHETIC. It is designed to look like real bank narrations
(merchant names, UPI/POS prefixes, reference numbers) and realistic spending
patterns (weekly rhythm, festival season, salary on the 1st), but model scores
obtained on it do not prove real-world accuracy.

Used by scripts/generate_dataset.py and by the tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ merchants
MERCHANTS = {
    "Food": ["Swiggy", "Zomato", "Dominos", "McDonalds", "KFC", "Starbucks", "Cafe Coffee Day",
             "Haldirams", "Burger King", "Pizza Hut", "Subway", "Chaayos", "BigBasket", "Blinkit",
             "Zepto", "DMart", "Kirana Store", "Barbeque Nation", "Behrouz Biryani", "Faasos",
             "Meghana Foods", "A2B Restaurant", "Theobroma", "Nature's Basket", "Udupi Hotel"],
    "Transport": ["Uber", "Ola", "Rapido", "IRCTC", "Indian Oil", "HP Petrol Pump", "Bharat Petroleum",
                  "Namma Metro", "Delhi Metro", "RedBus", "IndiGo", "Namma Yatri", "FASTag",
                  "Shell Fuel", "Yulu", "Air India", "KSRTC", "BluSmart", "Vogo", "MakeMyTrip Flights"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa", "Croma", "Reliance Digital",
                 "Decathlon", "Lifestyle", "Westside", "Meesho", "IKEA", "Tata Cliq", "H&M", "Zara",
                 "Pantaloons", "Bata", "Lenskart", "Vijay Sales", "Shoppers Stop"],
    "Bills": ["Airtel Postpaid", "Jio Recharge", "Vi Recharge", "BESCOM Electricity", "Tata Power",
              "ACT Fibernet", "Mahanagar Gas", "BWSSB Water", "Indane Gas", "Tata Play DTH",
              "BSNL Broadband", "Adani Electricity", "LIC Premium", "Hathway Broadband", "Excitel"],
    "Entertainment": ["Netflix", "Prime Video", "Spotify", "Disney Hotstar", "BookMyShow", "PVR Cinemas",
                      "INOX", "Steam Games", "PlayStation Store", "YouTube Premium", "Zee5", "SonyLIV",
                      "Wonderla", "Smaaash", "JioCinema", "Timezone Arcade"],
    "Education": ["Coursera", "Udemy", "Unacademy", "BYJUS", "University Fee", "NPTEL Exam",
                  "Sapna Book House", "Crossword Books", "LeetCode Premium", "GeeksforGeeks",
                  "upGrad", "Physics Wallah", "Exam Registration", "Library Membership"],
    "Healthcare": ["Apollo Pharmacy", "MedPlus", "Tata 1mg", "PharmEasy", "Practo", "Manipal Hospital",
                   "Fortis Hospital", "Dr Lal PathLabs", "Thyrocare", "Netmeds", "Aster Clinic",
                   "Narayana Health", "Dental Clinic", "Eye Care Centre"],
    "Rent": ["House Rent", "PG Rent", "NoBroker Rent", "Flat Rent", "Hostel Rent", "Landlord Rent",
             "Apartment Maintenance", "Housing Society"],
    "Other": ["ATM Cash Withdrawal", "UPI Transfer Ramesh", "UPI Transfer Priya", "Naturals Salon",
              "Laundry Service", "Blue Dart Courier", "Donation Akshaya Patra", "Gift Shop",
              "Bank Service Charge", "Dry Cleaning", "UPI Transfer Friend", "Temple Donation",
              "Barber Shop", "Pet Store", "Urban Company"],
}

# Category-specific context words a real bank narration often contains.
CONTEXT_WORDS = {
    "Food": ["order", "food delivery", "dine in", "groceries", "restaurant", "meal", "cafe"],
    "Transport": ["ride", "trip", "fuel", "ticket", "travel", "cab", "booking"],
    "Shopping": ["purchase", "order", "online shopping", "store", "apparel", "electronics"],
    "Bills": ["bill payment", "recharge", "utility", "monthly bill", "autopay"],
    "Entertainment": ["subscription", "movie tickets", "membership", "streaming", "games"],
    "Education": ["course fee", "tuition", "books", "exam fee", "certification"],
    "Healthcare": ["pharmacy", "medicines", "consultation", "lab test", "hospital bill"],
    "Rent": ["rent", "monthly rent", "rent payment", "maintenance"],
    "Other": ["transfer", "payment", "service", "misc"],
}

PREFIXES = ["", "", "UPI-", "POS ", "NEFT-", "IMPS/", "ECOM "]
CITIES = ["BANGALORE", "MUMBAI", "DELHI", "CHENNAI", "HYDERABAD", "PUNE", "KOLKATA"]

# Per category: (mean txns per month, median amount at the 50k-income reference, log-sd)
CATEGORY_PATTERN = {
    "Food":          (18, 320, 0.55),
    "Transport":     (10, 230, 0.60),
    "Shopping":      (3, 1500, 0.60),
    "Bills":         (4, 700, 0.45),
    "Entertainment": (3, 450, 0.55),
    "Education":     (0.6, 1800, 0.60),
    "Healthcare":    (1.0, 650, 0.65),
    "Rent":          (1, None, 0.0),   # fixed amount per profile
    "Other":         (3, 600, 0.60),
}

PROFILES = [
    # id, label, monthly income, rent, category multipliers
    {"profile_id": "P1", "label": "College student", "income": 18000, "rent": 6000,
     "mult": {"Education": 2.0, "Shopping": 0.6}},
    {"profile_id": "P2", "label": "Junior developer", "income": 38000, "rent": 11000, "mult": {}},
    {"profile_id": "P3", "label": "Mid-level professional", "income": 65000, "rent": 18000,
     "mult": {"Entertainment": 1.3}},
    {"profile_id": "P4", "label": "Senior manager", "income": 120000, "rent": 32000,
     "mult": {"Shopping": 1.4, "Transport": 1.3}},
    {"profile_id": "P5", "label": "Family household", "income": 85000, "rent": 24000,
     "mult": {"Healthcare": 2.0, "Education": 2.5, "Food": 1.4}},
    {"profile_id": "P6", "label": "Freelancer", "income": 52000, "rent": 14000,
     "mult": {"Other": 1.5}, "income_volatility": 0.25},
]


# ------------------------------------------------------------------ helpers
def make_description(category: str, rng: np.random.Generator, merchant: str | None = None) -> str:
    merchant = merchant or rng.choice(MERCHANTS[category])
    style = rng.integers(0, 6)
    ref = rng.integers(100000, 999999)
    context = rng.choice(CONTEXT_WORDS[category])
    if style == 0:
        return merchant
    if style == 1:
        return f"{merchant} {context}".title()
    if style == 2:
        return f"UPI-{merchant.upper().replace(' ', '')}-{ref}@ybl"
    if style == 3:
        return f"POS {merchant.upper()} {rng.choice(CITIES)}"
    if style == 4:
        return f"{rng.choice(PREFIXES)}{merchant} {context} #{ref}".strip()
    return f"{context.title()} - {merchant}"


def _amount(category: str, scale: float, mult: float, rng, rent: float) -> float:
    if category == "Rent":
        return float(rent)
    _, median, sd = CATEGORY_PATTERN[category]
    value = rng.lognormal(np.log(median * scale * mult), sd)
    return float(max(10.0, round(value / 5) * 5))


def generate_profile_transactions(profile: dict, start: str, months: int, rng: np.random.Generator,
                                  anomaly_rate: float = 0.0, monthly_growth: float = 0.003,
                                  discretionary_growth: float = 0.0) -> pd.DataFrame:
    """Generate `months` months of income + expense transactions for one profile.

    anomaly_rate: probability that an expense is replaced by an unusually large
    one (5-12x the typical amount). Those rows get is_anomaly = 1.
    discretionary_growth: extra monthly growth for Shopping/Entertainment/Food
    (used to build an 'overspending' scenario).
    """
    rows = []
    scale = profile.get("scale", (profile["income"] / 50000) ** 0.6)
    month_starts = pd.date_range(start=start, periods=months, freq="MS")
    # Spending level persists from month to month (lifestyle drift): an AR(1)
    # process on the log scale, level_t = 0.8 * level_{t-1} + noise.
    level = 0.0
    for m_idx, month_start in enumerate(month_starts):
        days = month_start.days_in_month
        trend = (1 + monthly_growth) ** m_idx
        level = 0.8 * level + rng.normal(0, 0.07)
        month_noise = float(np.exp(level))

        # ---- income on the 1st (freelancers vary)
        vol = profile.get("income_volatility", 0.03)
        salary = profile["income"] * rng.lognormal(0, vol)
        rows.append({"date": month_start, "description": rng.choice(
            ["Monthly Salary", "Salary Credit", "NEFT Salary ACME Corp", "Stipend Credit"]
            if profile["profile_id"] != "P6" else ["Client Payment", "Freelance Invoice Payment", "NEFT Client"]),
            "amount": round(salary, -1), "type": "Income", "category": "Income", "is_anomaly": 0})
        if month_start.month == 3 and profile["profile_id"] in {"P3", "P4"}:
            rows.append({"date": month_start + pd.Timedelta(days=14), "description": "Annual Bonus Credit",
                         "amount": round(profile["income"] * 0.5, -1), "type": "Income",
                         "category": "Income", "is_anomaly": 0})

        # ---- expenses
        for category, (freq, _, _) in CATEGORY_PATTERN.items():
            mult = profile["mult"].get(category, 1.0)
            season = 1.0
            if category == "Shopping" and month_start.month in (10, 11):
                season = 1.6          # Diwali / festive sales
            if category == "Entertainment" and month_start.month == 12:
                season = 1.3
            growth = (1 + discretionary_growth) ** m_idx if category in ("Shopping", "Entertainment", "Food") else 1
            if category == "Rent":
                n = 1
            else:
                n = rng.poisson(freq * (mult if category in ("Education", "Healthcare") else 1.0)
                                * (season if category == "Shopping" else 1.0))
            for _ in range(n):
                if category == "Rent":
                    day = 1 + int(rng.integers(0, 5))
                else:
                    # weekends are busier for food/entertainment
                    day = int(rng.integers(1, days + 1))
                    if category in ("Food", "Entertainment") and rng.random() < 0.3:
                        d = month_start + pd.Timedelta(days=day - 1)
                        day = min(days, day + (5 - d.dayofweek) % 7)
                date = month_start + pd.Timedelta(days=day - 1)
                amount = _amount(category, scale, mult, rng, profile["rent"])
                if category != "Rent":   # rent is a fixed contract amount
                    amount *= trend * month_noise * growth * (season if category != "Shopping" else 1.0)
                is_anom = 0
                if category != "Rent" and rng.random() < anomaly_rate:
                    amount *= rng.uniform(5, 12)
                    is_anom = 1
                rows.append({"date": date, "description": make_description(category, rng),
                             "amount": float(round(amount)), "type": "Expense",
                             "category": category, "is_anomaly": is_anom})

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.insert(0, "profile_id", profile["profile_id"])
    return df


def generate_training_corpus(seed: int = 42, start: str = "2023-10-01", months: int = 36,
                             anomaly_rate: float = 0.01) -> pd.DataFrame:
    """All profiles, 36 months, with labelled injected anomalies."""
    rng = np.random.default_rng(seed)
    frames = [generate_profile_transactions(p, start, months, rng, anomaly_rate=anomaly_rate) for p in PROFILES]
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "txn_id", [f"T{i:06d}" for i in range(1, len(df) + 1)])
    return df


# ------------------------------------------------------------------ splits
def split_classifier_data(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42):
    """Expense rows split BY MERCHANT, so test merchants never appear in training.

    This avoids description leakage (e.g. 'Swiggy' in both train and test) and
    measures how well the model generalises to unfamiliar merchants.
    """
    exp = df[df["type"] == "Expense"].copy()
    exp["merchant"] = exp.apply(lambda r: _find_merchant(r["description"], r["category"]), axis=1)
    rng = np.random.default_rng(seed)
    test_merchants = set()
    for category, merchants in MERCHANTS.items():
        k = max(1, int(round(len(merchants) * test_fraction)))
        test_merchants.update(rng.choice(merchants, size=k, replace=False))
    is_test = exp["merchant"].isin(test_merchants)
    cols = ["txn_id", "description", "category", "merchant"]
    return exp.loc[~is_test, cols].reset_index(drop=True), exp.loc[is_test, cols].reset_index(drop=True)


def _find_merchant(description: str, category: str) -> str:
    text = description.lower().replace(" ", "")
    for m in MERCHANTS[category]:
        if m.lower().replace(" ", "") in text:
            return m
    return "unknown"


def split_by_time(df: pd.DataFrame, train_until: str, val_until: str | None = None):
    """Chronological split: everything before `train_until` is training data."""
    dates = pd.to_datetime(df["date"])
    train = df[dates < train_until]
    if val_until is None:
        return train.reset_index(drop=True), df[dates >= train_until].reset_index(drop=True)
    val = df[(dates >= train_until) & (dates < val_until)]
    test = df[dates >= val_until]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ------------------------------------------------------------------ demo scenarios
DEMO_PROFILE = {"profile_id": "DEMO", "label": "Demo user", "income": 55000, "rent": 15000, "mult": {}}
OVERSPEND_PROFILE = {"profile_id": "DEMO", "label": "Demo user (overspending)", "income": 30000,
                     "rent": 7000, "scale": 0.55, "mult": {"Shopping": 1.4, "Entertainment": 1.5}}


def _to_upload_format(df: pd.DataFrame, rng, blank_fraction: float = 0.7) -> pd.DataFrame:
    """Turn generated rows into what a user would upload: most categories blank."""
    out = df[["date", "description", "amount", "type", "category"]].copy()
    out.insert(0, "Transaction ID", [f"D{i:05d}" for i in range(1, len(out) + 1)])
    blank = (out["type"] == "Expense") & (rng.random(len(out)) < blank_fraction)
    out.loc[blank, "category"] = ""
    return out.rename(columns={"date": "Date", "description": "Description", "amount": "Amount",
                               "type": "Type", "category": "Category"})


def generate_demo_scenarios(seed: int = 7) -> dict[str, pd.DataFrame]:
    """Three demo CSVs + a generic sample. Same generator, different behaviour."""
    start, months = "2026-04-01", 6
    rng = np.random.default_rng(seed)
    normal = generate_profile_transactions(DEMO_PROFILE, start, months, rng, anomaly_rate=0.0,
                                           monthly_growth=0.0)
    # Scenario 2: same history + two unusual high-value transactions in the last month
    anomaly = normal.copy()
    extra = pd.DataFrame([
        {"profile_id": "DEMO", "date": "2026-09-18", "description": "Zomato Order - Party Catering",
         "amount": 6800.0, "type": "Expense", "category": "Food", "is_anomaly": 1},
        {"profile_id": "DEMO", "date": "2026-09-23", "description": "POS CROMA ELECTRONICS BANGALORE",
         "amount": 38500.0, "type": "Expense", "category": "Shopping", "is_anomaly": 1},
    ])
    anomaly = pd.concat([anomaly, extra], ignore_index=True).sort_values("date").reset_index(drop=True)

    rng_o = np.random.default_rng(seed + 1)
    overspend = generate_profile_transactions(OVERSPEND_PROFILE, start, months, rng_o, anomaly_rate=0.0,
                                              monthly_growth=0.0, discretionary_growth=0.05)

    rng_u = np.random.default_rng(seed + 2)
    return {
        "demo_normal.csv": _to_upload_format(normal, rng_u),
        "demo_anomaly.csv": _to_upload_format(anomaly, rng_u),
        "demo_overspending.csv": _to_upload_format(overspend, rng_u),
    }
