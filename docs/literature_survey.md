# Literature Survey and Domain Analysis (Member 1 – Research Lead)

> **How to use this file.** The structure and the references are ready. Every reference below is a real, citable publication; check each DOI/URL once yourself and read at least the abstract before quoting it. The "existing systems" section describes products in general terms. Product features change, so verify current details on each official website before the review and add the access date.

## 1. Problem statement
Traditional expense trackers record transactions and draw summary charts. Users still have to work out three things by themselves: (a) which expenses are unusual, (b) how much they are likely to spend next month, and (c) how to adjust their budget to reach a savings goal. FinGuard AI automates all three. It categorises transactions with ML, flags abnormal spending with anomaly detection, forecasts next-month spending, and uses a goal-based intelligent agent that turns these signals into a feasible revised budget, persisted as alerts and proposals.

## 2. Motivation
- Financial literacy is linked to better saving and retirement planning, and many people lack it (Lusardi & Mitchell, 2014).
- Behavioural economics shows that people find it hard to save without help; automatic, default-based plans raise saving rates (Thaler & Benartzi, 2004).
- Digital payments in India (UPI) produce long, noisy transaction histories that are hard to review by hand. The RBI's Account Aggregator framework (2016 Master Direction) shows regulators expect consented, data-driven financial services.

## 3. Domain concepts (for the report)
| Term | Meaning in FinGuard |
|---|---|
| Income | Money received in a month (salary, client payments) |
| Expense | Money spent; split into 9 categories |
| Essential vs discretionary expense | Essential (rent, bills, education, healthcare by default) cannot be cut; discretionary (shopping, entertainment, ...) can |
| Disposable income | Income − essential expenses |
| Monthly budget / spending limit | Maximum allowed spending = income − savings goal (or a user cap, if lower) |
| Savings goal | Amount the user wants to keep each month |
| Overspending | Spending above the limit |

## 4. Related work

### 4.1 Anomaly detection
- Chandola, Banerjee & Kumar (2009) survey anomaly detection techniques: statistical, distance-based, density-based, and so on. Their survey frames our choice of an unsupervised method when labels are rare.
- Liu, Ting & Zhou (2008, 2012) introduced **Isolation Forest**: anomalies are isolated by fewer random splits. It has linear time complexity, needs no distance computations and no labels. This is our Component 1.
- Breunig et al. (2000) proposed the Local Outlier Factor (LOF), a density-based alternative. We mention it as a possible comparison.
- Bolton & Hand (2002) review statistical fraud detection in banking and highlight unsupervised methods for spotting unusual behaviour, which is closely related to unusual personal spending.

### 4.2 Text classification of transactions
- Salton & Buckley (1988) established TF-IDF term weighting, which we use to turn narrations into features.
- Logistic regression on TF-IDF features is a standard, strong baseline for short-text classification. It is implemented in scikit-learn (Pedregosa et al., 2011).

### 4.3 Forecasting
- Hyndman & Athanasopoulos (2021) explain time-based evaluation and why simple benchmarks (mean, naïve) must always be reported.
- Makridakis, Spiliotis & Assimakopoulos (2018) show that simple statistical methods often match or beat complex ML on small series. This matches our finding that Ridge only slightly beats the historical mean.
- Hoerl & Kennard (1970) introduced ridge regression (L2-regularised least squares), which suits small, correlated feature sets such as our lags.

### 4.4 Intelligent agents
- Russell & Norvig (2021) define agent types (simple reflex, model-based, goal-based, utility-based) and the PEAS description we use.
- Wooldridge (2009) covers agent architectures and the perceive–decide–act cycle.

## 5. Existing systems (verify current features before presenting)
| System type | Examples | Typical capability | Gap that FinGuard addresses |
|---|---|---|---|
| Manual trackers | Spreadsheets, "Money Manager"-style apps | Manual entry, charts | No automatic categorisation, anomaly detection or planning |
| Rule-based budgeting apps | YNAB (envelope / zero-based budgeting) | User-defined budgets, rule categorisation | Rules must be written by the user; no prediction of unusual spending |
| SMS/aggregator trackers (India) | Walnut (now axio), bank apps | Auto-import from SMS / bank, rule categories | Mostly descriptive; limited explainable goal planning |
| Discontinued mainstream tool | Intuit Mint (shut down in 2024; users were moved to Credit Karma) | Aggregation, categories, budgets | Shows the need for transparent, locally-controlled tools |

**Existing vs proposed**
| Feature | Traditional tracker | FinGuard AI |
|---|---|---|
| Categorisation | Manual / keyword rules | TF-IDF + Logistic Regression with confidence and user correction |
| Unusual expenses | Not detected | Isolation Forest score + statistical explanation |
| Future spending | Not estimated | Ridge forecast with honest fallback |
| Budget advice | Static | Goal-based agent: feasibility check, constrained cuts, reasons |
| Follow-up | None | Persistent alerts, proposals, review flags, auto-resolution |
| Privacy | Cloud | Fully local (SQLite), no API keys |

## 6. Scope
**In scope:** a single user, CSV import, nine expense categories, monthly planning, simulated actions.
**Out of scope:** real bank connections, money movement, investment advice, multi-currency, authentication.

## 7. Real-world applications and limitations
Applications: personal budgeting, student finance education, a pre-screening step for bank "insights" features, family expense monitoring. Limitations: synthetic training data, Indian merchant vocabulary, rule-based cut caps, and forecasts with about 13 % error.

## References (APA style)
1. Bolton, R. J., & Hand, D. J. (2002). Statistical fraud detection: A review. *Statistical Science, 17*(3), 235–255. https://doi.org/10.1214/ss/1042727940
2. Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. *Proceedings of ACM SIGMOD 2000*, 93–104. https://doi.org/10.1145/342009.335388
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys, 41*(3), Article 15. https://doi.org/10.1145/1541880.1541882
4. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634
5. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
6. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM)*, 413–422. https://doi.org/10.1109/ICDM.2008.17
7. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2012). Isolation-based anomaly detection. *ACM Transactions on Knowledge Discovery from Data, 6*(1), Article 3. https://doi.org/10.1145/2133360.2133363
8. Lusardi, A., & Mitchell, O. S. (2014). The economic importance of financial literacy: Theory and evidence. *Journal of Economic Literature, 52*(1), 5–44. https://doi.org/10.1257/jel.52.1.5
9. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and machine learning forecasting methods: Concerns and ways forward. *PLOS ONE, 13*(3), e0194889. https://doi.org/10.1371/journal.pone.0194889
10. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
11. Reserve Bank of India. (2016, updated). *Master Direction – Non-Banking Financial Company – Account Aggregator (Reserve Bank) Directions, 2016*. https://www.rbi.org.in
12. Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern approach* (4th ed.). Pearson.
13. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0
14. Thaler, R. H., & Benartzi, S. (2004). Save More Tomorrow™: Using behavioral economics to increase employee saving. *Journal of Political Economy, 112*(S1), S164–S187. https://doi.org/10.1086/380085
15. Wooldridge, M. (2009). *An introduction to multiagent systems* (2nd ed.). Wiley.
