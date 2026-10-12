# Data Dictionary and Dataset Description

## 1. Upload CSV format

| Column (accepted names) | Required | Type / format | Notes |
|---|---|---|---|
| Transaction ID (`txn_id`, `id`, `reference`) | optional | text | Used for duplicate detection. If missing, a deterministic fingerprint `AUTO-YYYYMMDD-xxxxxxxx` is created from date + description + amount + type. |
| Date (`transaction date`, `value date`) | **yes** | `YYYY-MM-DD`, `DD-MM-YYYY`, `DD/MM/YYYY`, `YYYY/MM/DD`, `DD Mon YYYY` | Unparseable dates → row dropped with the line number reported |
| Description (`narration`, `particulars`, `merchant`) | **yes** | text | Input to the classifier |
| Amount (`amount (inr)`) | **yes** | positive number; `₹`, `Rs.`, `INR` and commas allowed | ≤ 0 or non-numeric → row dropped |
| Type (`transaction type`, `dr/cr`) | **yes** | Income/Expense; also Credit/Debit, CR/DR, Deposit/Withdrawal | Others → row dropped |
| Category | optional | one of the 9 categories (case-insensitive) or blank | Blank/unknown → predicted by ML. Income rows always get `Income`. |

Expense categories: Food, Transport, Shopping, Bills, Entertainment, Education, Healthcare, Rent, Other.

Duplicate rules: same Transaction ID (or the same date+description+amount+type when there is no ID) inside a file is skipped. A row whose ID already exists in the database is skipped. Both are reported to the user.

## 2. Generated datasets (`data/generated/`, all SYNTHETIC, seed 42)

| File | Rows | Content | Used for |
|---|---|---|---|
| `transactions_full.csv` | 9,861 | 6 profiles × 36 months (2023-10 → 2026-09), columns `txn_id, profile_id, date, description, amount, type, category, is_anomaly` | Source of all others |
| `classifier_train.csv` | 7,196 | expense descriptions + category + merchant | Classifier training |
| `classifier_test.csv` | 2,443 | 38 merchants never seen in training | Classifier test |
| `anomaly_dataset.csv` | 9,639 | expenses with `split` = train (until 2025-03, 4,788) / val (2025-04 → 09, 1,584) / test (2025-10 →, 3,267) and label `is_anomaly` | Isolation Forest fit / contamination choice / test |
| `monthly_totals.csv` | 216 | `profile_id, month, expense, regular_expense, split` (test = 2025-10 →) | Forecaster |
| `dataset_summary.json` | – | counts and splits | Documentation |

Demo / upload files in `data/`: `demo_normal.csv` (264 rows), `demo_anomaly.csv` (266), `demo_overspending.csv` (275), each covering April to September 2026 with about 70 % of the expense categories left blank; `sample_transactions.csv` (25 rows with one duplicate, one bad date and one bad amount).

### How the data is generated (`src/data_generator.py`)
- **Profiles:** student (₹18k income), junior developer (₹38k), mid-level professional (₹65k), senior manager (₹120k), family (₹85k), freelancer (₹52k, variable income).
- **Transactions per month:** a Poisson number per category (e.g. food ≈ 18, transport ≈ 10, shopping ≈ 3). Amounts are log-normal around a category median scaled by income^0.6. Rent is a fixed amount.
- **Descriptions:** about 150 real-world Indian merchant names written in six bank-narration styles (`UPI-SWIGGY-123456@ybl`, `POS CROMA BANGALORE`, `Fuel - Indian Oil`, ...).
- **Realistic patterns:** weekend shift for food and entertainment; festive Oct/Nov shopping ×1.6; December entertainment ×1.3; March bonus for two profiles; a persistent month-to-month spending level (AR(1) with coefficient 0.8), because real spending habits drift rather than reset each month.
- **Labelled anomalies:** each expense has a 1 % chance of being multiplied by 5–12× (`is_anomaly = 1`). This gives 101 anomalies in total: 42 in train, 18 in val, 41 in test.

## 3. Observations (expense data)

| Category | Count | Median ₹ | Mean ₹ | Max ₹ |
|---|---|---|---|---|
| Food | 3,854 | 377 | 507 | 11,834 |
| Transport | 2,202 | 264 | 388 | 8,581 |
| Bills | 830 | 784 | 981 | 17,048 |
| Shopping | 707 | 1,681 | 2,341 | 36,156 |
| Entertainment | 693 | 512 | 670 | 10,125 |
| Other | 688 | 713 | 963 | 19,172 |
| Healthcare | 271 | 1,154 | 1,542 | 7,606 |
| Rent | 216 | 16,000 | 17,500 | 32,000 |
| Education | 178 | 2,792 | 4,080 | 50,784 |

- The class imbalance is strong: food alone is 40 % of expenses, education under 2 %. Hence `class_weight="balanced"` and macro-F1 as a metric.
- Amounts are heavily right-skewed (mean > median everywhere), so the anomaly model uses log amounts.
- Rent is few in number but dominates the budget, and it is an essential category, so the agent never cuts it.
- Month totals vary by about ±15 % even without anomalies, which limits how accurate any forecast can be.

## 4. Database tables (`finguard.db`)

| Table | Purpose | Key columns |
|---|---|---|
| `transactions` | All imported transactions + ML outputs | `txn_id` (unique), `category`, `category_source` (user/model/corrected), `category_confidence`, `needs_review`, `anomaly_score`, `is_anomaly`, `typical_amount`, `anomaly_explanation`, `review_status` (NONE/PENDING_REVIEW/REVIEWED) |
| `settings` | User inputs | `monthly_income`, `savings_goal`, `monthly_budget`, `essential_categories` (JSON) |
| `alerts` | Persistent alerts | `alert_key` (unique), `alert_type`, `severity`, `status` (ACTIVE/ACKNOWLEDGED/RESOLVED), `resolved_reason` |
| `budget_proposals` | Revised budgets | `target_month`, `status` (PROPOSED/SUPERSEDED/ACCEPTED), `feasibility`, `plan_json`, `plan_hash` |
| `recommendations` | Savings advice | `rec_key` (unique), `kind`, `message`, `status` |
| `action_log` | Audit trail of every automated action | `run_id`, `action`, `target`, `outcome` |
| `analysis_runs` | Snapshot of each analysis (state, assessment, plan) | `run_id`, `result_json` |

## 5. Limitations of the data
- It is synthetic. Real statements contain abbreviations, P2P transfers with personal names, refunds and reversals, and multiple accounts.
- Anomalies are a single type: a multiplied amount. Real fraud and real one-off expenses are more varied.
- The six profiles do not represent the whole population. Real-world evaluation would need consented, anonymised user data. Public alternatives exist (for example Kaggle personal-finance datasets), but they differ in schema and licence and would have to be checked before use.
