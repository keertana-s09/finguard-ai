# Review 3 — Final Results, Contributions, Risks, Test Cases

All numbers come from `reports/evaluation_report.md` (produced by `python scripts/evaluate_models.py` on synthetic data, seed 42). Re-run the script before the review and copy the numbers it prints.

## 1. Actual evaluation metrics

| Component | Result | Baseline | Verdict |
|---|---|---|---|
| Categorisation (2,443 test rows, 38 unseen merchants) | Accuracy 0.747, macro-F1 0.715 | Keywords 0.659 / 0.688 | ML better; weak on Entertainment and Shopping brand names |
| Low-confidence review | 47.5 % of test rows flagged; accuracy 0.953 confident vs 0.520 flagged | – | The confidence threshold is meaningful |
| Isolation Forest (3,267 test expenses, 41 anomalies) | P 0.812, R 0.634, F1 0.712, 32 flags | 3× rule: P 0.296, R 0.976, F1 0.455, 135 flags | Far fewer false alerts; misses about a third |
| One-hot category ablation | F1 0.478 | – | Justifies the chosen feature design |
| Forecast (72 profile-months) | MAE ₹6,539 (13.3 %) | Mean ₹6,574 · MA-3 ₹6,810 | Only marginally better than the mean |
| Agent (500 random cases) | 7 / 7 checks pass in 500 / 500 | Independent re-computation | Constraints always respected |

Demo scenarios: Normal → ON_TRACK; Anomaly → 2 HIGH alerts + 2 review flags; Overspending → ACHIEVABLE (Shopping −₹927); Goal ₹15,000 → INFEASIBLE (realistic goal ≈ ₹11,389).

## 2. Contributions of all five members
See [team_contributions.md](team_contributions.md). On a slide, show one row per member: modules owned, deliverables, and one thing they can demo.

## 3. Possible failures and alternative solutions

| Risk | How it shows | What the system does now | Alternative / future work |
|---|---|---|---|
| Unfamiliar merchants | Low confidence or wrong category | Confidence < 0.60 → "needs review"; user correction is stored and never overwritten | Retrain periodically on user corrections; merchant dictionary; sentence embeddings |
| Insufficient history | Fewer than 3 months or fewer than 10 expenses | Labelled rule-based forecast; anomaly detection skipped with a message; reference medians for new categories | Population priors, Bayesian updating |
| False anomaly alerts | Rare but legitimate expense flagged | Explanation shown; **Confirm as legitimate** → alert auto-resolved and not raised again | Per-user threshold learned from confirmations |
| Unrealistic savings goal | Goal > feasible savings | Agent returns INFEASIBLE, maximum-cut plan, realistic goal; no fake plan | Multi-month plans, income-side suggestions |
| Poor forecast quality | ~13 % error | Baselines reported; anomalies excluded; method labelled | Prediction intervals, more history, category-level models |
| Invalid CSV | Wrong columns / dates / amounts | Clear errors, row-level warnings with line numbers, nothing partially broken | Column-mapping UI |
| Model–library mismatch on a new laptop | Pickle load error | App retrains automatically | Pin versions in `requirements.txt` |

## 4. Working demo test cases

| # | Action | Expected result |
|---|---|---|
| T1 | Upload `data/sample_transactions.csv` | 22 imported; warnings: 1 duplicate, 1 bad date, 1 bad amount; categories predicted |
| T2 | Upload the same file again | 0 imported, "already existed" warning |
| T3 | Upload a CSV without an Amount column | Error "Missing required column(s): amount" |
| T4 | Load scenario 1 | Agent ON_TRACK, no unusual transactions, savings recommendation |
| T5 | Load scenario 2 | 2 flagged (₹6,800 food, ₹38,500 shopping), HIGH alerts, review flags |
| T6 | Scenario 2 → Confirm the Croma transaction | Its alert becomes RESOLVED |
| T7 | Load scenario 3 | ACHIEVABLE_WITH_CUTS, Shopping −₹927, proposal saved |
| T8 | Scenario 3 → set the goal to 15,000 | INFEASIBLE alert; the previous overspending alert is RESOLVED |
| T9 | Press "Re-run AI analysis" twice | Action log shows UNCHANGED; no new alerts |
| T10 | Correct a category | Source becomes `corrected`; it stays after re-run |
| T11 | Alerts → Acknowledge | Status ACKNOWLEDGED; it stays acknowledged after re-run |
| T12 | Export → Download report | Markdown report with summary, plan, alerts |

Automated: `python -m pytest -v` → 46 tests (all 13 required scenarios; mapping in README §6).

## 5. Final documentation checklist
README · architecture · workflow · ML explanation · data dictionary · literature survey · team contributions · user manual · viva Q&A · poster · evaluation report · screenshots (`docs/screenshots/`).
