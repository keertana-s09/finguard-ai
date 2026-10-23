# Viva Questions and Answers

## Project & integration
**Q1. What are your three AI components?**
Anomaly detection (Isolation Forest), a goal-based intelligent agent (`FinancialDecisionAgent`), and automated response (`AutomatedResponseEngine`). Classification and forecasting are supporting ML.

**Q2. Show that the components are really connected.**
`pipeline.run_analysis()`: the detector's output goes into (a) the forecaster, as one-offs excluded from its inputs, and (b) `agent.perceive()`. The agent's `generate_plan()` produces action dicts, and `execute_actions()` passes them to the engine, which writes to SQLite. The dashboard reads SQLite. `test_anomaly_to_alert_to_dashboard_chain` checks this chain automatically.

**Q3. Why is this an agent and not just if-else code?**
It has an explicit goal (spending ≤ income − savings goal), an internal state, and it evaluates hypothetical future budgets (planned spending minus candidate cuts) before choosing actions. It can also conclude that the goal is unreachable. A simple reflex agent maps the current condition to an action without any goal or look-ahead.

**Q4. What kind of environment is it?**
Partially observable (the future is predicted), stochastic, sequential (stored alerts and reviews affect later runs), static during one decision, discrete actions, single agent.

**Q5. Is the automated response "real"?**
Yes. It inserts, updates and resolves rows in four tables, flags transactions, deduplicates by key, supersedes old proposals and logs every action. It does not move money, by design.

## Isolation Forest
**Q6. How does Isolation Forest work?**
It builds many random trees, each splitting on a random feature at a random value. Anomalies are few and different, so they are isolated near the root (short path). Score ≈ 2^(−average path length / c(n)).

**Q7. What is contamination and how did you choose it?**
It is the expected fraction of anomalies, which sets the flagging threshold. We tried 0.005–0.05 and chose 0.01 by F1 on the validation months only, never on the test months.

**Q8. Why not one-hot encode the category?**
We tested it: validation and test F1 dropped (0.71 → 0.48). Random feature selection wastes splits on mostly-zero columns. The category enters through "excess over the category's typical amount" instead.

**Q9. Does Isolation Forest explain why a transaction is anomalous?**
No. It gives a score. Our explanation compares the amount with the category median from the user's history. It is a separate statistical computation, and the UI says so.

**Q10. Why is the rule baseline's recall higher?**
The synthetic anomalies are 5–12× the typical amount, which is exactly what the 3× rule tests. But it raises 135 alerts against our 32, so its precision is 0.30 versus 0.81.

## Classification
**Q11. What is TF-IDF?**
Term frequency × inverse document frequency: words frequent in one description but rare overall get high weight. We use word 1–2-grams and character 3–5-grams.

**Q12. How did you avoid leakage?**
The split is by merchant (test merchants never appear in training), and digits/reference numbers are removed.

**Q13. What happens with a merchant the model has never seen?**
Its confidence is usually low, so it is marked for review. The user's correction is saved and never overwritten.

## Forecasting
**Q14. Why Ridge?**
It suits small data, its L2 regularisation stabilises correlated lag features, and it is explainable.

**Q15. Your Ridge MAE is almost the same as the average. Is the model useless?**
It is only marginally better (₹6,539 vs ₹6,574). Monthly totals here are mostly random. Literature such as Makridakis et al. (2018) shows that simple benchmarks are hard to beat on short series. We report this honestly instead of claiming a big gain.

**Q16. How do you prevent training on the future?**
The split is time-based, all lag features use `shift`, and a unit test checks that changing a later month does not change earlier features.

**Q17. What if the user has only two months of data?**
The app uses a rule-based average, labelled "RULE-BASED", and does not pretend it is ML.

## Agent details
**Q18. How are cuts chosen?**
The gap is forecast minus limit. Candidates are the discretionary categories, ordered by: anomaly this month first, then cut priority, then growth. Each takes min(remaining gap, cap × planned). Essentials always have a cap of 0.

**Q19. Where do the caps (50 % shopping, 25 % food, …) come from?**
They are planning assumptions written in `config.py`, not learned. They can be changed, and a learned version is future work.

**Q20. What if income is not given?**
It is detected as the average income over the last 3 months. With zero income the goal is INFEASIBLE.

## Data & testing
**Q21. Is your data real?**
No, it is synthetic and reproducible. Results show the pipeline works; they do not prove real-world accuracy.

**Q22. How do you prevent duplicate alerts?**
Each alert has a unique key (type + month or transaction). Same key and same message → UNCHANGED. The DB enforces UNIQUE on the key.

**Q23. How many tests and what do they cover?**
55 pytest tests covering preprocessing, ML, agent, response engine, end-to-end scenarios, a headless start-up of the Streamlit app, and the FastAPI layer used by the optional React frontend.
