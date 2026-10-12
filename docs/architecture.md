# System Architecture

All diagrams are Mermaid (render on GitHub, in VS Code with the "Markdown Preview Mermaid Support" extension, or at https://mermaid.live).

## 1. Layered architecture

```mermaid
flowchart TD
    subgraph L1[Presentation layer]
        UI[Streamlit dashboard<br/>app.py + charts.py]
    end
    subgraph L2[Input & data layer]
        IN[CSV upload / demo scenario / user settings]
        PRE[Data validation & preprocessing<br/>preprocessing.py]
        DB[(SQLite database<br/>database.py)]
    end
    subgraph L3[ML processing layer]
        CLF[Transaction classification<br/>TF-IDF + Logistic Regression]
        AD[Anomaly detection - COMPONENT 1<br/>Isolation Forest]
        FC[Monthly spending forecast<br/>Ridge Regression]
    end
    subgraph L4[Decision layer]
        ST[Financial state aggregation<br/>FinancialDecisionAgent.perceive]
        AG[Goal-based intelligent agent - COMPONENT 2<br/>evaluate_goals + generate_plan]
    end
    subgraph L5[Action layer]
        AR[Automated Response Engine - COMPONENT 3<br/>automated_response.py]
        A1[Persistent spending alerts]
        A2[Budget proposals]
        A3[Transaction review flags]
        A4[Savings recommendations]
    end
    UI --> IN --> PRE --> DB
    DB --> CLF --> AD --> FC
    AD -->|one-off anomalies excluded from inputs| FC
    CLF --> ST
    AD --> ST
    FC --> ST
    ST --> AG -->|action requests| AR
    AR --> A1 & A2 & A3 & A4 --> DB
    DB --> UI
```

`src/pipeline.py` (`FinGuardPipeline.run_analysis`) executes these layers in this exact order. The dashboard, the demo scenarios and the integration tests all call the same method.

## 2. Integration of the three AI components (one run)

```mermaid
sequenceDiagram
    participant U as User / Dashboard
    participant P as FinGuardPipeline
    participant C as ExpenseClassifier
    participant I as AnomalyDetector (Isolation Forest)
    participant F as SpendingForecaster (Ridge)
    participant A as FinancialDecisionAgent
    participant R as AutomatedResponseEngine
    participant D as SQLite

    U->>P: upload CSV / load scenario / change settings
    P->>D: insert validated transactions
    P->>C: predict categories (blank rows only)
    C-->>D: category, confidence, needs_review
    P->>I: score expenses
    I-->>D: anomaly_score, is_anomaly, explanation
    P->>F: forecast regular spending (one-off anomalies excluded)
    P->>A: perceive(transactions, forecast, anomalies)
    A->>A: evaluate_goals() -> ON_TRACK / ACHIEVABLE / INFEASIBLE
    A->>A: generate_plan() -> cuts, revised budget, action list
    A->>R: execute_actions(actions)
    R->>D: create/update alerts (dedupe by key)
    R->>D: flag transactions PENDING_REVIEW
    R->>D: save budget proposal (old one SUPERSEDED)
    R->>D: auto-resolve alerts whose condition disappeared
    R->>D: write every outcome to action_log
    U->>D: dashboard reads alerts, proposals, log
```

### Concrete example (demo scenario 2)
1. A ₹38,500 "POS CROMA ELECTRONICS" expense is inserted.
2. Isolation Forest gives it the highest anomaly score (0.109) and flags it; the statistics module explains: *"20.6× the typical amount (₹1,869)"*.
3. The forecaster drops it from the inputs (one-off), so next month's forecast is not inflated.
4. The agent perceives it as a high-side anomaly in the current month: ₹38,500 ≥ 10 % of income → **HIGH** severity. It also sees that September spending (₹89,792) exceeds the limit (₹46,127), mostly because of unusual transactions.
5. The agent requests `FLAG_TRANSACTION`, `CREATE_ALERT(UNUSUAL_TRANSACTION)` and `CREATE_ALERT(CURRENT_MONTH_OVER_BUDGET)`.
6. The response engine writes them to SQLite. A second run produces `UNCHANGED` (no duplicates).
7. The user clicks **Confirm as legitimate** → next run: the agent no longer requests the alert → the engine resolves it automatically.

## 3. Agent architecture (goal-based)

```mermaid
flowchart LR
    E[Environment<br/>transactions, ML outputs, settings] -->|percepts| P[perceive]
    P --> S[(Internal state<br/>FinancialState)]
    S --> G[evaluate_goals<br/>limit = income - goal<br/>gap = forecast - limit]
    GOAL[[Goal: next-month spending <= limit<br/>essentials untouched, cuts <= caps]] --> G
    G --> D{status}
    D -->|ON_TRACK| PL[generate_plan]
    D -->|ACHIEVABLE| PL
    D -->|INFEASIBLE| PL
    PL --> ACT[action list]
    ACT -->|execute_actions| R[Automated Response Engine]
    R -->|changes stored state| E
```

## 4. Database schema

```mermaid
erDiagram
    TRANSACTIONS {
        int id PK
        text txn_id UK
        text date
        text description
        real amount
        text type
        text category
        text category_source
        real category_confidence
        int needs_review
        real anomaly_score
        int is_anomaly
        text anomaly_explanation
        text review_status
    }
    ALERTS {
        int id PK
        text alert_key UK
        text alert_type
        text severity
        text status
        int related_txn FK
    }
    BUDGET_PROPOSALS {
        int id PK
        text target_month
        text status
        text feasibility
        text plan_json
        text plan_hash
    }
    RECOMMENDATIONS {
        int id PK
        text rec_key UK
        text kind
        text status
    }
    ACTION_LOG {
        int id PK
        text run_id
        text action
        text outcome
    }
    ANALYSIS_RUNS {
        int id PK
        text run_id UK
        text result_json
    }
    SETTINGS {
        text key PK
        text value
    }
    TRANSACTIONS ||--o{ ALERTS : "related_txn"
    ANALYSIS_RUNS ||--o{ ACTION_LOG : "run_id"
```

## 5. Module responsibilities

| Module | Responsibility | Key functions / classes |
|---|---|---|
| `preprocessing.py` | Validate & clean CSV, monthly totals | `load_csv`, `validate_transactions`, `clean_description` |
| `database.py` | SQLite schema and all queries | `FinanceDB` |
| `expense_classifier.py` | Categorisation + keyword baseline | `ExpenseClassifier`, `KeywordBaselineClassifier` |
| `anomaly_detector.py` | Isolation Forest + statistical explanations | `AnomalyDetector`, `high_side_anomalies` |
| `spending_predictor.py` | Ridge forecast + baselines + fallback | `SpendingForecaster`, `make_lag_features` |
| `financial_agent.py` | Goal-based agent | `FinancialDecisionAgent.perceive / evaluate_goals / generate_plan / execute_actions` |
| `automated_response.py` | Executes and persists actions, auto-resolves | `AutomatedResponseEngine.execute` |
| `pipeline.py` | Orchestration, demo scenarios | `FinGuardPipeline.run_analysis`, `DEMO_SCENARIOS` |
| `training.py` | Training, contamination selection, model loading | `train_all`, `load_models` |
| `reports.py`, `charts.py` | Export, visualisation | `markdown_report`, chart builders |

## 6. Hardware / software requirements
- Any laptop with 4 GB RAM and a dual-core CPU; ~400 MB disk for the virtual environment. No GPU.
- Python 3.11+, Windows 10/11, macOS or Linux. Runs fully offline after `pip install`.
- Training all models takes about a minute; one analysis run on a few hundred transactions takes about a second.
