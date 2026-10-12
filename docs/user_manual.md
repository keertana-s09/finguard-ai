# FinGuard AI — User Manual

## Installing and starting
See README §1. In short: create a virtual environment, run `pip install -r requirements.txt`, then `streamlit run app.py`, and open http://localhost:8501.

## Pages

**🏠 Home Dashboard.** Six cards: income, this month's expenses, remaining balance, predicted next month, savings goal, unusual transactions. Charts: income vs expenses, spending by category, actual vs predicted, savings gauge (black line = goal), recommended budget, active alerts.

**📤 Upload Transactions.**
- *Option A* – load one of four demo scenarios. This **replaces** all current data.
- *Option B* – upload your CSV. Tick "Clear existing data" to start fresh; otherwise new rows are added and duplicates are skipped. Download the sample CSV to see the format.
- Required columns: Date, Description, Amount, Type. Optional: Transaction ID, Category.

**🏷️ Expense Classification.** Each expense shows its category, where the category came from (`user` = your file, `model` = ML, `corrected` = you fixed it), and the model's confidence. Toggle "Show only low-confidence predictions" to see what needs checking. Use **Correct a category** to fix one; the whole analysis re-runs.

**🚨 Anomaly Detection.** Grey dots = normal expenses, red diamonds = unusual ones (log scale). Each flagged transaction shows its score, its typical amount and an explanation. **Confirm as legitimate** marks it reviewed and resolves its alert.

**📈 Spending Prediction.** Shows the predicted spending for next month and the method. "ML (Ridge)" needs at least 3 complete months; otherwise you get a clearly labelled rule-based average. The chart compares actual spending (all and regular) with the model's predictions for past months.

**🤖 AI Financial Advisor.** Set:
- *Monthly income*: 0 means it is detected from your income transactions.
- *Savings goal*.
- *Monthly spending cap* (optional).
- *Essential categories*: the agent never cuts these.

Press **Save and run the agent**. The page shows the four agent steps: what it perceived, the goal check (on track / achievable / infeasible), the plan with reasons, and the actions executed.

**🔔 Alerts and Reports.**
- *Alerts*: acknowledge active alerts; resolved alerts are kept for history.
- *Budget proposals*: accept the latest proposal (simulated, no money moves).
- *Recommendations*: current savings advice.
- *Action log*: every automated action and its outcome.
- *Export*: Markdown report and CSV of all transactions with ML outputs.

**Sidebar.** Transaction count, active alerts, agent status, and **Re-run AI analysis**.

## Common messages
| Message | Meaning / fix |
|---|---|
| "Missing required column(s)" | Rename your columns to Date, Description, Amount, Type |
| "rows dropped – unreadable date" | Use YYYY-MM-DD or DD/MM/YYYY |
| "already existed in the database" | Re-upload detected; nothing was duplicated |
| "RULE-BASED forecast" | Fewer than 3 complete months; add more history |
| "Anomaly detection skipped" | Fewer than 10 expenses |
| "reference-data medians were used" | A category had little history; its typical amount comes from the training data |

## Resetting
Upload page → **Clear all data**, or delete the `finguard.db` file while the app is stopped.

## Privacy
Everything stays on your computer in `finguard.db`. The app makes no network calls and has no access to bank accounts.
