# Review 2 — Evidence Pack

## 1. System / block architecture
Use the diagrams in [architecture.md](architecture.md): the layered architecture, the integration sequence and the database schema. Render them at https://mermaid.live and export PNGs for the slides.

## 2. Dataset description and accessibility
- **Source:** our own reproducible generator (`python scripts/generate_dataset.py`, seed 42). No private or paid data; anyone can regenerate it.
- **Size:** 9,861 transactions, 6 financial profiles, 36 months (Oct 2023 – Sep 2026), 9 expense categories, about 150 Indian merchants, 101 labelled anomalies.
- **Splits:** classifier split by merchant (7,196 / 2,443 rows); anomaly split by time (train / val / test); forecast split by time (test = last 12 months).
- **Demo files:** 3 scenarios + a messy sample upload.
- Details: [data_dictionary.md](data_dictionary.md). State clearly in the slides that the data is synthetic.

## 3. Benchmark and evaluation plan
| Component | Metric | Baseline | Protocol |
|---|---|---|---|
| Categorisation | Accuracy, macro-F1, confusion matrix | Keyword rules | Merchant-disjoint test set |
| Anomaly detection | Precision, recall, F1, number of flags | "amount > 3× typical" rule | Fit on train months, tune contamination on val months, report on test months |
| Forecast | MAE, MAPE | Historical mean; 3-month moving average | Time-based split, no shuffling |
| Agent | Feasibility correctness, constraint satisfaction, action correctness | Independent re-computation | 500 random situations + 4 demo scenarios |

Run: `python scripts/evaluate_models.py` → `reports/evaluation_report.md`.

## 4. Baseline approaches
- `KeywordBaselineClassifier`: about 100 hand-written keywords. Most hits wins, otherwise "Other".
- `rule_baseline`: flag when the amount is more than 3× the category's typical amount.
- `baseline_historical_mean`, `baseline_moving_average`: average of all previous months / of the last three months.

## 5. Technology stack and hardware
| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for ML; the team knows it |
| UI | Streamlit | Interactive web app in pure Python |
| ML | scikit-learn | TF-IDF, Logistic Regression, Isolation Forest, Ridge; reproducible pipelines |
| Data | Pandas, NumPy | Tabular processing |
| Charts | Plotly | Interactive charts with hover |
| Storage | SQLite (built in) | Zero-install persistent database |
| Model files | joblib | Saves fitted pipelines |
| Testing | pytest + Streamlit AppTest | Automated unit, integration and UI start-up tests |
| IDE | VS Code, Windows Terminal | |

Hardware: any laptop with 4 GB RAM; no GPU; ~400 MB of disk for packages; works offline.

## 6. Initial working prototype — demo steps
1. `streamlit run app.py`.
2. Upload Transactions → load **Scenario 3 (Overspending)**.
3. Expense Classification: show the predicted categories and confidence bars; correct one category.
4. Anomaly Detection: show the scatter plot and the flagged transactions.
5. Spending Prediction: forecast ₹23,927 for 2026-10 with the actual-vs-predicted chart.
6. AI Financial Advisor: status *Achievable with cuts*, Shopping −₹927, options table.
7. Alerts and Reports: the alert, the budget proposal and the action log.
8. Change the goal to ₹15,000 → *Goal infeasible*; the old alert is resolved automatically.
