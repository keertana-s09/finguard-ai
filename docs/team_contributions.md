# Team Contributions and Integration Plan

Team of five, using the faculty-defined roles. Each member **owns** some modules (they write and explain them) and **reviews** another member's modules. All five run the full test suite and practise the demo.

## 1. Roles

| Member | Faculty role | Owns (code / docs) | Reviews | Key deliverables |
|---|---|---|---|---|
| **M1** | Research Lead – literature survey & domain | `docs/literature_survey.md`, problem statement, motivation, domain concepts, existing-vs-proposed table, `src/config.py` (essential categories, cut caps: the finance assumptions) | M5 report, agent rules (finance correctness) | Literature survey with verified references, domain analysis, scope |
| **M2** | AI Analyst – AI concepts | `expense_classifier.py`, `anomaly_detector.py`, `spending_predictor.py`, `financial_agent.py`, `training.py`, `scripts/train_models.py`, `scripts/evaluate_models.py`, `docs/ml_explanation.md` | M3 data splits (leakage), M4 integration | Trained models, evaluation report, confusion matrix, forecast evaluation, agent architecture & PEAS |
| **M3** | Data Lead – data collection & observation | `data_generator.py`, `preprocessing.py`, `scripts/generate_dataset.py`, all CSVs, `docs/data_dictionary.md`, `tests/test_preprocessing.py` | M2 feature engineering | Train/test datasets, demo CSVs, sample upload, data dictionary, observations |
| **M4** | Prototype Lead – demo & implementation | `app.py`, `charts.py`, `database.py`, `automated_response.py`, `pipeline.py`, `reports.py`, `tests/test_automated_response.py`, `tests/test_integration.py`, screenshots | M2 agent actions | Working dashboard, integrated backend, demo scenarios, screenshots |
| **M5** | Documentation & Presentation | README, `docs/architecture.md`, `docs/workflow.md`, review decks (`review1/2/3_content.md`), `docs/user_manual.md`, `docs/viva_questions.md`, `docs/poster_content.md`, project report | Everyone's docstrings & READMEs | Review 1–3 PPTs, report, poster, user manual, viva Q&A, demo coordination |

`tests/test_ml_pipeline.py` and `tests/test_agent.py` are owned by M2 and reviewed by M1, who checks that the finance logic makes sense.

## 2. Integration points (who must agree on what)

| Interface | Producer → consumer | Contract |
|---|---|---|
| Clean transaction DataFrame | M3 `validate_transactions` → M4 `insert_transactions` | columns `txn_id, date (YYYY-MM-DD), description, amount (>0), type (Income/Expense), category (or NaN)` |
| Classifier output | M2 → M4 `classify()` | `predicted_category, confidence, needs_review` |
| Anomaly output | M2 → M2 agent / M4 DB | `anomaly_score, is_anomaly, typical_amount, anomaly_explanation` per expense `id` |
| Forecast | M2 → agent | `ForecastResult.to_dict()` (`target_month, predicted, method, note, history`) |
| Agent actions | M2 agent → M4 engine | list of dicts with `action` ∈ {CREATE_ALERT, FLAG_TRANSACTION, SAVE_BUDGET_PROPOSAL, CREATE_RECOMMENDATION} |
| Run result | M4 pipeline → M4 UI / M5 docs | `analysis_runs.result_json` (state, assessment, plan, action_results) |

## 3. Suggested timeline (12 weeks)

| Week | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| 1–2 | Survey, problem statement | AI component selection, PEAS | CSV schema, generator v1 | Repo, SQLite schema | Review 1 deck |
| 3–4 | Existing systems table | Classifier + baseline | Splits, preprocessing tests | Upload page, DB layer | Review 1 rehearsal |
| 5–6 | Finance rules (caps, essentials) | Isolation Forest, forecaster | Demo scenarios | Pipeline + response engine | Architecture diagrams |
| 7–8 | Review agent rules | Agent + evaluation script | Data observations | All dashboard pages | Review 2 deck |
| 9–10 | Limitations, applications | Tuning on validation only | Data dictionary | Integration tests, screenshots | Report draft, user manual |
| 11–12 | Final references check | Final metrics | Final datasets | Demo freeze | Review 3 deck, poster, viva Q&A |

## 4. Shared responsibilities (all members)
- Run `python -m pytest -v` and `python scripts/evaluate_models.py` on your own laptop before each review.
- Each member should be able to explain the full chain: CSV → classifier → Isolation Forest → forecast → agent → response engine → dashboard.
- Each member presents at least one demo scenario at Review 3.
- Use `docs/viva_questions.md` for mutual question practice.

## 5. Contribution evidence for Review 3
Keep a short weekly log (date, task, files changed). If you use Git, the commit history shows contributions automatically: `git log --author="name" --stat`.
