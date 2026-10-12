# Poster Content (A1 portrait, 3 columns)

**Title:** FinGuard AI — Intelligent Personal Finance Management and Budget Optimization Agent
**Subtitle:** Anomaly Detection + Goal-Based Agent + Automated Response · Dept. of CSE · Team M1–M5 · Guide

### Column 1
**Problem.** Expense apps record spending but do not warn about unusual expenses, predict next month, or plan a realistic budget.

**Objectives**
1. Categorise transactions with ML (beat keyword rules).
2. Detect unusual expenses (Isolation Forest).
3. Forecast next-month spending (Ridge).
4. Plan a feasible budget and persist alerts automatically (goal-based agent + response engine).

**Data.** Synthetic, reproducible: 9,861 transactions, 6 profiles, 36 months, 101 labelled anomalies.

### Column 2 — Architecture (use the layered diagram from docs/architecture.md)
CSV → validation → SQLite → classifier → **Isolation Forest** → forecast → **Goal-based agent** → **Automated response** → dashboard

**Agent (PEAS):** goal = spending ≤ income − savings goal; essentials are never cut; cuts are capped; infeasible goals are reported honestly.

### Column 3 — Results (synthetic test data)
| | Ours | Baseline |
|---|---|---|
| Category accuracy (unseen merchants) | 0.747 | 0.659 |
| Anomaly F1 | 0.712 | 0.455 |
| Anomaly false flags | 32 | 135 |
| Forecast MAE | ₹6,539 | ₹6,574 |
| Agent constraint checks | 500/500 | – |

**Demo:** ₹38,500 purchase → flagged → HIGH alert + review flag → user confirms → alert auto-resolved.

**Limitations & future work:** real data validation, learning cut caps from user behaviour, prediction intervals.

**QR code:** link to the repository / demo video.
