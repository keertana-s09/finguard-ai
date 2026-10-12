# Review 1 — Presentation Content (8 slides)

> Review 1 is about the problem and the design. Do **not** show final accuracy numbers here. Results belong to Review 2/3.

---

### Slide 1 — Title
**FinGuard AI: Intelligent Personal Finance Management and Budget Optimization Agent**
Team members (M1–M5) · Guide · Department of CSE · Domain: Finance
AI combination No. 10: **Anomaly Detection + Intelligent Agent + Automated Response**

*Speaker (M5):* "We are building an assistant that not only records spending, but notices unusual expenses, predicts next month and plans a budget to reach a savings goal."

---

### Slide 2 — Problem Statement
- Expense apps **record** transactions but do not **decide** anything.
- Users cannot easily spot unusual expenses in hundreds of UPI/card transactions.
- They do not know next month's likely spending.
- They do not know **which** categories to cut, or whether their savings goal is even realistic.

**Problem:** Build a system that automatically categorises transactions, detects abnormal spending, predicts upcoming expenses and generates a feasible, explained budget plan with persistent alerts.

*Speaker (M1):* "The gap is between data and decisions."

---

### Slide 3 — Motivation
- Low financial literacy is linked to poor saving (Lusardi & Mitchell, 2014).
- Automatic, default-based plans increase saving (Thaler & Benartzi, 2004), so automation helps.
- UPI has made transaction histories long and noisy; manual review does not scale.
- Unusual expenses (fraud, impulse purchases, billing errors) are costly if noticed late.

*Speaker (M1):* "A good assistant should warn early and suggest a realistic fix."

---

### Slide 4 — Why AI?
| Need | Plain rules | AI approach |
|---|---|---|
| Categorise "UPI-ZEPTO-88123@ybl" | Keyword list breaks on new merchants | TF-IDF + Logistic Regression learns patterns and gives a confidence |
| Detect unusual spending | One fixed threshold for everyone | Isolation Forest learns what is rare from the data, without labels |
| Predict next month | Copy last month | Ridge regression on recent months + seasonality |
| Decide what to cut | Static advice | Goal-based agent evaluates options against a goal and constraints |

*Speaker (M2):* "Rules are still used where they belong, such as the essential-category constraints. Learning is used where patterns change."

---

### Slide 5 — Objectives (measurable)
1. **Classification:** categorise expense descriptions into 9 categories and beat a keyword-rule baseline on accuracy and macro-F1, measured on merchants not seen in training.
2. **Anomaly detection:** flag unusually high expenses with Isolation Forest and report precision, recall and F1 on labelled held-out months, compared with a simple threshold rule.
3. **Forecasting:** predict next-month spending with a time-based split, and report MAE against historical-average and moving-average baselines.
4. **Budget decisions:** the agent must (a) correctly classify goals as on-track / achievable / infeasible, (b) never cut essential categories, (c) keep every cut within its cap, and (d) persist each decision exactly once. We will verify all four on a set of test scenarios.

---

### Slide 6 — Agent Architecture: Goal-Based Agent
```
 Environment ──percepts──► perceive() ──► Internal state (FinancialState)
      ▲                                         │
      │                              evaluate_goals()  ◄── Goal: spending ≤ income − savings goal
      │                                         │
      └── automated response ◄── actions ◄── generate_plan()
```
- **Environment:** the user's transactions, ML outputs, settings.
- **Percepts:** income, current spending per category, forecast, anomalies, savings goal, essential categories.
- **Internal state:** `FinancialState` (current month, baseline per category, anomalies with review status).
- **Goal:** next-month spending ≤ limit, essentials untouched, cuts within limits.
- **Decision mechanism:** compute the gap; evaluate the maximum feasible cut per category; greedy prioritised allocation; detect infeasibility.
- **Actions:** create alert, flag transaction, save budget proposal, create recommendation.

**PEAS**
| P | Goal met, constraints respected, no impossible plans, no duplicate alerts |
|---|---|
| E | Transactions, ML outputs, settings (partially observable, stochastic, sequential) |
| A | Alert / flag / proposal / recommendation requests |
| S | Income, category spending, forecast, anomaly scores, goal, essentials |

*Speaker (M2):* "It is goal-based because it compares possible future budgets with an explicit goal before acting. A reflex agent would only react to the current number."

---

### Slide 7 — AI Components and Integration
**Anomaly Detection → Intelligent Agent → Automated Response**
1. **Isolation Forest** scores each expense (features: log amount, excess over the category's typical amount, day of week).
2. The **agent** reads the anomalies: it raises alert severity, cuts the affected category first, and excludes one-offs from the forecast.
3. The **response engine** stores alerts, review flags and budget proposals in SQLite, prevents duplicates and auto-resolves alerts when the situation changes.

Example chain: ₹38,500 electronics purchase → Isolation Forest flags it → agent requests a HIGH alert and a review flag → engine stores both → dashboard shows the alert → user confirms the purchase → next run resolves the alert.

Supporting ML (not counted as components): TF-IDF + Logistic Regression, Ridge Regression.

---

### Slide 8 — Plan, Stack, Team
- **Stack:** Python 3.11+, Streamlit, scikit-learn, Pandas/NumPy, Plotly, SQLite, joblib, pytest. Free, offline, no GPU.
- **Data:** a reproducible synthetic generator (6 profiles × 36 months) with labelled anomalies. We will clearly state that it is synthetic.
- **Team:** M1 Research · M2 AI · M3 Data · M4 Prototype · M5 Documentation.
- **Next (Review 2):** architecture diagram, dataset, baselines, first working prototype.

---

## Likely Review 1 questions
- *Why not an LLM?* There is no need: decisions must be transparent, reproducible and free of API costs.
- *Why is the agent not just if-else?* It keeps state, evaluates alternative future budgets against a goal, and proves infeasibility. Each decision has a reason.
- *Why Isolation Forest?* It needs no labels, is fast, and works on a few features.
