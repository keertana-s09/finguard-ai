# AI Components and ML Explanation

## 0. Which parts count as AI components?

| Faculty component | Implementation | File |
|---|---|---|
| **1. Anomaly Detection** | Isolation Forest on expense transactions | `src/anomaly_detector.py` |
| **2. Intelligent Agent** | Goal-based agent `FinancialDecisionAgent` | `src/financial_agent.py` |
| **3. Automated Response** | `AutomatedResponseEngine`: persistent alerts, proposals, review flags, auto-resolution | `src/automated_response.py` |
| Supporting ML (not counted as a separate component) | TF-IDF + Logistic Regression categorisation; Ridge Regression forecast | `expense_classifier.py`, `spending_predictor.py` |

Python, Streamlit, Pandas, SQLite and Plotly are tools. We do not count them as AI components.

---

## 1. Expense categorisation — TF-IDF + Logistic Regression

**Pipeline** (`build_pipeline`):
1. `clean_description`: lower-case the text and remove digits and symbols, e.g. `UPI-SWIGGY-8812331@ybl → "upi swiggy ybl"`. Reference numbers carry no category information.
2. `FeatureUnion` of two TF-IDF vectorisers:
   - word 1–2-grams catch context words such as *ride*, *pharmacy*, *bill payment*;
   - character 3–5-grams catch partial merchant names such as *pharm*, *petro*, *cinema*.
3. Multinomial `LogisticRegression(C=5, class_weight="balanced")`, which gives a probability per category.

**Confidence & review.** Confidence is the highest class probability. Below 0.60 the transaction gets `needs_review = 1`. On the test set the model is 95.3 % accurate on confident predictions and 52.0 % on flagged ones, so the threshold separates reliable predictions from unreliable ones.

**User labels win.** Categories given in the CSV (`category_source = user`) or corrected in the UI (`corrected`) are never overwritten.

**Evaluation protocol.** Merchant-disjoint split: 25 % of the merchants in each category appear only in the test set (38 merchants). This prevents description leakage and measures how the model handles unfamiliar merchants.

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.747 | 0.715 |
| Keyword-rule baseline (`KeywordBaselineClassifier`) | 0.659 | 0.688 |

Weak classes are Entertainment (F1 0.50) and Shopping (0.53). Unseen brand names such as "Zee5" or "Lifestyle" carry no context word. This is the "unfamiliar merchants" risk, and the low-confidence review flag is the mitigation.

The deployed model (`models/expense_classifier.joblib`) is refitted on all labelled rows after evaluation.

---

## 2. Anomaly detection — Isolation Forest (Component 1)

**Idea.** Isolation Forest (Liu, Ting & Zhou, 2008) builds random trees that split the data at random points. Unusual points are isolated after only a few splits, so a short average path length means a high anomaly score. It needs no labels.

**Input.** Expense transactions only (income is excluded). Transaction IDs are never features.

**Features** (`build_feature_matrix`):

| Feature | Meaning |
|---|---|
| `log_amount` | log(1 + amount). This is how the transaction amount enters; the raw amount is too skewed. |
| `excess_over_typical` | max(0, log(amount / typical)), where *typical* is the median of the user's **previous** (up to 60) transactions in the same category. This is how the **category** and the **spending history** enter. It is one-sided because only overspending matters. |
| `day_of_week` | 0 = Monday … 6 = Sunday |

**Insufficient history.** If a category has fewer than 5 earlier transactions, *typical* falls back to the reference median learned from the training data, and the UI says so. With fewer than 10 categorised expenses in total, detection is skipped with a clear message.

**Design decision backed by an experiment.** We first tried one-hot category columns. On the validation months they *lowered* F1, because Isolation Forest picks features at random, so nine mostly-zero columns dilute the useful ones. The evaluation script keeps this as an ablation:

| Method (test months, 41 labelled anomalies) | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| Isolation Forest (used) | 0.812 | 0.634 | 0.712 | 32 |
| + one-hot category (ablation) | 0.615 | 0.390 | 0.478 | 26 |
| Rule: amount > 3× typical | 0.296 | 0.976 | 0.455 | 135 |

**Contamination** (expected anomaly share) = 0.01. It was chosen from {0.005, 0.01, 0.02, 0.03, 0.05} by F1 on the validation months only. `random_state=42` makes the results reproducible.

**Score & explanation.** `anomaly_score = −decision_function`, so higher means more unusual. The sentence *"Your ₹6,800 food transaction is 18.5× the typical amount (₹368, median of your previous food transactions)"* is computed by `explain()` from category statistics. **Isolation Forest does not produce explanations**, and the UI states this.

**Honest caveat.** The injected test anomalies are 5–12× the typical amount, and the 3× rule is built to catch exactly that, which gives it near-perfect recall. Isolation Forest wins on F1 because it raises about four times fewer false alerts.

---

## 3. Spending forecast — Ridge Regression

**Target.** Next month's **regular** spending: the monthly total of expenses minus the transactions flagged as unusually high (≥ 2× typical). One-off purchases are unpredictable and would inflate the next forecast. This is the second integration point: the anomaly detector feeds the forecaster.

**Features** for month *t* (from months before *t* only):
- `hist_mean`: average of all months before *t*, the user's normal level;
- `lag_1_rel, lag_2_rel, lag_3_rel`: previous three months ÷ `hist_mean`;
- `mean_3_rel`: average of those three ÷ `hist_mean`;
- `month_sin, month_cos`: calendar month (festive season).

**Target** = y / `hist_mean`, and the prediction is multiplied back by `hist_mean`. Working with ratios makes the model scale-free, so one model trained on six profiles (₹18k student … ₹120k manager) can be applied to any user. Model: `StandardScaler → Ridge(alpha=1)`.

**Leakage control.** The split is time-based (train: target months up to 2025-09; test: 2025-10 … 2026-09). Months are never shuffled, and lags are computed with `shift`. `test_lag_features_never_use_future_months` checks this automatically.

**Fallback.** With fewer than 3 complete months, the app shows a **rule-based** forecast (average of the available months), labelled "RULE-BASED". An incomplete last month (last transaction before day 25) is excluded from the inputs.

| Method (72 test profile-months) | MAE | MAPE |
|---|---|---|
| Ridge Regression | ₹6,539 | 13.3 % |
| Historical average baseline | ₹6,574 | 13.3 % |
| 3-month moving average baseline | ₹6,810 | 14.1 % |

**Interpretation.** Ridge is only marginally better than the long-run average. In this data most month-to-month variation comes from a few random large purchases, which no model can foresee. We report this honestly and do not claim a large improvement. Ridge was kept because it is simple, explainable (its coefficients can be shown), and learns the festive-season effect.

Development note: Ridge on raw rupee amounts was *worse* than both baselines because the profiles differ about 4× in scale. The ratio formulation fixed that.

---

## 4. Goal-based intelligent agent (Component 2)

### PEAS
| | |
|---|---|
| **Performance measure** | Next-month planned spending ≤ limit; savings goal met; essential categories untouched; cuts within caps; no impossible plans; alerts not duplicated |
| **Environment** | The user's transactions, ML outputs (categories, anomaly scores, forecast), user settings. Partially observable (future spending is estimated), stochastic, sequential (each run builds on stored alerts and reviews), single-agent |
| **Actuators** | Action requests: `CREATE_ALERT`, `FLAG_TRANSACTION`, `SAVE_BUDGET_PROPOSAL`, `CREATE_RECOMMENDATION` (executed by the response engine) |
| **Sensors** | `perceive()`: income (user or detected), current-month spending by category, 3-month baseline per category (one-offs removed), forecast, high-side anomalies with review status, savings goal, budget cap, essential categories |

### Why goal-based (and not just reflex rules)
A simple reflex agent maps "spending high → alert". Our agent holds an explicit **goal** (spending ≤ income − savings goal, plus the optional cap) and an **internal state**. It **evaluates future states**: the predicted spending under different cut combinations. It then chooses actions that reach the goal, or proves the goal unreachable.

### Decision mechanism
1. `limit = min(income − savings_goal, budget_cap)`.
2. `planned[c] = forecast × share of c in the recent baseline`.
3. For each category: `max_cut[c] = planned[c] × cap[c]` (0 for essentials). Caps: Entertainment/Shopping 50 %, Other 40 %, Food 25 %, Transport 20 %. These are planning assumptions in `config.py`, not learned values.
4. `gap = forecast − limit`
   - gap ≤ 0 → **ON_TRACK**
   - gap ≤ Σ max_cut → **ACHIEVABLE_WITH_CUTS**
   - otherwise → **INFEASIBLE** (also when income is 0).
5. Greedy constrained allocation: candidates sorted by (has an anomaly this month → first), then cut priority, then growth versus baseline. Each takes `min(remaining gap, max_cut)`. Every cut carries a reason string.
6. INFEASIBLE → apply the maximum cuts and report `max_achievable_savings` as a realistic goal, instead of an impossible plan.
7. Action rules: overspending alert (MEDIUM, or HIGH if the gap is > 10 % of income); infeasible-goal alert plus realistic-goal recommendation; month-over-limit alert (mentions how much of the excess is due to anomalies); one review flag per unreviewed high-side anomaly; alert for anomalies in the current month (HIGH if ≥ 10 % of income); always a budget proposal; one recommendation per cut, or a savings recommendation when on track.

### How anomalies influence the agent
- Categories with an anomaly this month are cut first, and the reason says so.
- Anomalies raise alert severity.
- Anomalies are removed from the baseline shares and from the forecast inputs (one-offs).
- An anomaly the user has confirmed (`REVIEWED`) is no longer alerted on.

### Agent evaluation
500 random situations (`evaluate_models.py`); all 7 checks pass in 500/500: feasibility verdict matches an independent re-computation, goal met when feasible, essentials untouched, caps respected, no cuts when on track, correct alert type, proposal saved.

---

## 5. Automated response (Component 3)

| Action | Persistent effect | Duplicate / change handling |
|---|---|---|
| `CREATE_ALERT` | Row in `alerts` (ACTIVE) | Same key + same text → `UNCHANGED`; changed text → `UPDATED`; resolved → re-activated; ACKNOWLEDGED kept |
| `FLAG_TRANSACTION` | `transactions.review_status = PENDING_REVIEW` | Skipped if the user already reviewed it |
| `SAVE_BUDGET_PROPOSAL` | Row in `budget_proposals` | Same plan hash → `UNCHANGED`; new plan → previous one becomes `SUPERSEDED` |
| `CREATE_RECOMMENDATION` | Row in `recommendations` | Upsert by key |
| (sync after each run) | Alerts, recommendations and review flags no longer requested → `RESOLVED` / cleared | Automatic status update when information changes |

Every outcome is written to `action_log`. Nothing touches a real account; proposals are simulated plans the user can accept.

---

## 6. Limitations
- Synthetic data: real narrations are messier (abbreviations, regional languages, P2P transfers).
- Single user per database; no authentication.
- The cut caps and priorities are expert assumptions; the system does not learn them from the user's behaviour.
- The forecast is only slightly better than an average, and the uncertainty is large (MAPE ≈ 13 %).
- Isolation Forest flags rare but legitimate expenses (e.g. annual insurance). The review/confirm flow mitigates this.
