"""SQLite storage for transactions, settings, alerts, budget proposals,
recommendations, the automated-action log and analysis runs.

A new connection is opened for every call; this keeps the class safe to use
from Streamlit (which re-runs the script on different threads).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_DB_PATH
from src.utils import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id              TEXT UNIQUE NOT NULL,
    date                TEXT NOT NULL,
    description         TEXT NOT NULL,
    amount              REAL NOT NULL,
    type                TEXT NOT NULL CHECK (type IN ('Income', 'Expense')),
    category            TEXT,
    category_source     TEXT,        -- 'user' | 'model' | 'corrected' | 'rule'
    predicted_category  TEXT,
    category_confidence REAL,
    needs_review        INTEGER DEFAULT 0,  -- low-confidence category
    anomaly_score       REAL,
    is_anomaly          INTEGER DEFAULT 0,
    typical_amount      REAL,
    anomaly_explanation TEXT,
    review_status       TEXT DEFAULT 'NONE', -- NONE | PENDING_REVIEW | REVIEWED
    review_reason       TEXT,
    created_at          TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_key   TEXT UNIQUE NOT NULL,   -- dedupe key, e.g. FORECAST_OVERSPEND:2026-10
    alert_type  TEXT NOT NULL,
    severity    TEXT NOT NULL,          -- LOW | MEDIUM | HIGH
    title       TEXT NOT NULL,
    message     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ACTIVE',  -- ACTIVE | ACKNOWLEDGED | RESOLVED
    related_txn INTEGER,
    created_at  TEXT,
    updated_at  TEXT,
    resolved_reason TEXT
);
CREATE TABLE IF NOT EXISTS budget_proposals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    target_month  TEXT NOT NULL,
    status        TEXT NOT NULL,        -- PROPOSED | SUPERSEDED | ACCEPTED
    feasibility   TEXT NOT NULL,
    income        REAL,
    savings_goal  REAL,
    spending_limit REAL,
    predicted_spending REAL,
    revised_total REAL,
    plan_json     TEXT NOT NULL,
    plan_hash     TEXT NOT NULL,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_key    TEXT UNIQUE NOT NULL,
    kind       TEXT NOT NULL,
    message    TEXT NOT NULL,
    amount     REAL,
    status     TEXT NOT NULL DEFAULT 'ACTIVE',  -- ACTIVE | RESOLVED
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS action_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT,
    action     TEXT NOT NULL,
    target     TEXT,
    outcome    TEXT NOT NULL,   -- CREATED | UPDATED | UNCHANGED | RESOLVED | FLAGGED | SKIPPED
    detail     TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS analysis_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT UNIQUE NOT NULL,
    created_at TEXT,
    result_json TEXT NOT NULL
);
"""

TXN_COLUMNS = ["txn_id", "date", "description", "amount", "type", "category"]


class FinanceDB:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = str(path)
        self.init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def query_df(self, sql: str, params=()) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    # ================================================================ transactions
    def insert_transactions(self, df: pd.DataFrame) -> tuple[int, int]:
        """Insert validated rows. Returns (inserted, skipped_as_duplicate).

        Duplicates against rows already in the database are detected through
        the UNIQUE txn_id constraint.
        """
        inserted = 0
        created = now_iso()
        with self.connect() as conn:
            for row in df[TXN_COLUMNS].itertuples(index=False):
                category = None if pd.isna(row.category) else row.category
                source = "user" if category else None
                cur = conn.execute(
                    """INSERT OR IGNORE INTO transactions
                       (txn_id, date, description, amount, type, category, category_source, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(row.txn_id), row.date, row.description, float(row.amount),
                     row.type, category, source, created),
                )
                inserted += cur.rowcount
        return inserted, len(df) - inserted

    def get_transactions(self) -> pd.DataFrame:
        return self.query_df("SELECT * FROM transactions ORDER BY date, id")

    def count_transactions(self) -> int:
        with self.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    def update_classification(self, rows: pd.DataFrame):
        """rows: id, category, category_source, predicted_category, category_confidence, needs_review"""
        with self.connect() as conn:
            conn.executemany(
                """UPDATE transactions SET category=?, category_source=?, predicted_category=?,
                   category_confidence=?, needs_review=? WHERE id=?""",
                [(r.category, r.category_source, r.predicted_category, float(r.category_confidence),
                  int(r.needs_review), int(r.id)) for r in rows.itertuples(index=False)],
            )

    def update_anomalies(self, rows: pd.DataFrame):
        """rows: id, anomaly_score, is_anomaly, typical_amount, anomaly_explanation"""
        with self.connect() as conn:
            conn.execute("UPDATE transactions SET anomaly_score=NULL, is_anomaly=0, "
                         "typical_amount=NULL, anomaly_explanation=NULL")
            conn.executemany(
                """UPDATE transactions SET anomaly_score=?, is_anomaly=?, typical_amount=?,
                   anomaly_explanation=? WHERE id=?""",
                [(float(r.anomaly_score), int(r.is_anomaly),
                  None if pd.isna(r.typical_amount) else float(r.typical_amount),
                  r.anomaly_explanation, int(r.id)) for r in rows.itertuples(index=False)],
            )

    def correct_category(self, txn_pk: int, category: str):
        """User correction: overrides the model and clears the low-confidence flag."""
        with self.connect() as conn:
            conn.execute("UPDATE transactions SET category=?, category_source='corrected', needs_review=0 "
                         "WHERE id=?", (category, int(txn_pk)))

    def set_review_status(self, txn_pk: int, status: str, reason: str | None = None):
        with self.connect() as conn:
            if reason is None:
                conn.execute("UPDATE transactions SET review_status=? WHERE id=?", (status, int(txn_pk)))
            else:
                conn.execute("UPDATE transactions SET review_status=?, review_reason=? WHERE id=?",
                             (status, reason, int(txn_pk)))

    def get_review_status(self, txn_pk: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT review_status FROM transactions WHERE id=?", (int(txn_pk),)).fetchone()
            return row[0] if row else None

    # ================================================================ settings
    def get_settings(self) -> dict:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def save_settings(self, settings: dict):
        with self.connect() as conn:
            conn.executemany("INSERT INTO settings(key, value) VALUES(?, ?) "
                             "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                             [(k, json.dumps(v)) for k, v in settings.items()])

    # ================================================================ alerts
    def get_alert(self, alert_key: str):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM alerts WHERE alert_key=?", (alert_key,)).fetchone()
            return dict(row) if row else None

    def insert_alert(self, alert: dict):
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO alerts (alert_key, alert_type, severity, title, message, status,
                   related_txn, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)""",
                (alert["key"], alert["alert_type"], alert["severity"], alert["title"], alert["message"],
                 alert.get("related_txn"), ts, ts),
            )

    def update_alert(self, alert_key: str, **fields):
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.connect() as conn:
            conn.execute(f"UPDATE alerts SET {cols} WHERE alert_key=?", (*fields.values(), alert_key))

    def list_alerts(self, status: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM alerts"
        params = ()
        if status:
            sql += " WHERE status=?"
            params = (status,)
        return self.query_df(sql + " ORDER BY CASE severity WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END, "
                                   "updated_at DESC", params)

    def open_alerts_of_types(self, types: list[str]) -> list[dict]:
        marks = ",".join("?" * len(types))
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM alerts WHERE status != 'RESOLVED' AND alert_type IN ({marks})",
                                types).fetchall()
        return [dict(r) for r in rows]

    # ================================================================ budget proposals
    def latest_proposal(self, target_month: str | None = None):
        sql = "SELECT * FROM budget_proposals WHERE status != 'SUPERSEDED'"
        params = ()
        if target_month:
            sql += " AND target_month=?"
            params = (target_month,)
        with self.connect() as conn:
            row = conn.execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone()
        if not row:
            return None
        out = dict(row)
        out["plan"] = json.loads(out.pop("plan_json"))
        return out

    def insert_proposal(self, p: dict) -> int:
        with self.connect() as conn:
            conn.execute("UPDATE budget_proposals SET status='SUPERSEDED' WHERE target_month=? "
                         "AND status='PROPOSED'", (p["target_month"],))
            cur = conn.execute(
                """INSERT INTO budget_proposals (target_month, status, feasibility, income, savings_goal,
                   spending_limit, predicted_spending, revised_total, plan_json, plan_hash, created_at)
                   VALUES (?, 'PROPOSED', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p["target_month"], p["feasibility"], p["income"], p["savings_goal"], p["spending_limit"],
                 p["predicted_spending"], p["revised_total"], json.dumps(p["plan"]), p["plan_hash"], now_iso()),
            )
            return cur.lastrowid

    def set_proposal_status(self, proposal_id: int, status: str):
        with self.connect() as conn:
            conn.execute("UPDATE budget_proposals SET status=? WHERE id=?", (status, int(proposal_id)))

    def list_proposals(self) -> pd.DataFrame:
        return self.query_df("SELECT id, target_month, status, feasibility, income, savings_goal, spending_limit, "
                             "predicted_spending, revised_total, created_at FROM budget_proposals ORDER BY id DESC")

    # ================================================================ recommendations
    def upsert_recommendation(self, rec_key: str, kind: str, message: str, amount: float) -> str:
        ts = now_iso()
        with self.connect() as conn:
            row = conn.execute("SELECT message, amount, status FROM recommendations WHERE rec_key=?",
                               (rec_key,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO recommendations (rec_key, kind, message, amount, status, created_at, "
                             "updated_at) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)",
                             (rec_key, kind, message, amount, ts, ts))
                return "CREATED"
            if row["message"] == message and row["status"] == "ACTIVE":
                return "UNCHANGED"
            conn.execute("UPDATE recommendations SET message=?, amount=?, status='ACTIVE', updated_at=? "
                         "WHERE rec_key=?", (message, amount, ts, rec_key))
            return "UPDATED"

    def resolve_recommendations_except(self, keep_keys: list[str]) -> int:
        marks = ",".join("?" * len(keep_keys)) or "''"
        with self.connect() as conn:
            cur = conn.execute(f"UPDATE recommendations SET status='RESOLVED', updated_at=? "
                               f"WHERE status='ACTIVE' AND rec_key NOT IN ({marks})", (now_iso(), *keep_keys))
            return cur.rowcount

    def list_recommendations(self, status: str | None = "ACTIVE") -> pd.DataFrame:
        if status:
            return self.query_df("SELECT * FROM recommendations WHERE status=? ORDER BY id DESC", (status,))
        return self.query_df("SELECT * FROM recommendations ORDER BY id DESC")

    # ================================================================ action log + runs
    def log_action(self, run_id: str, action: str, target: str, outcome: str, detail: str = ""):
        with self.connect() as conn:
            conn.execute("INSERT INTO action_log (run_id, action, target, outcome, detail, created_at) "
                         "VALUES (?, ?, ?, ?, ?, ?)", (run_id, action, target, outcome, detail, now_iso()))

    def list_actions(self, limit: int = 200) -> pd.DataFrame:
        return self.query_df("SELECT * FROM action_log ORDER BY id DESC LIMIT ?", (limit,))

    def save_run(self, run_id: str, result: dict):
        with self.connect() as conn:
            conn.execute("INSERT INTO analysis_runs (run_id, created_at, result_json) VALUES (?, ?, ?)",
                         (run_id, now_iso(), json.dumps(result, default=str)))

    def latest_run(self):
        with self.connect() as conn:
            row = conn.execute("SELECT result_json FROM analysis_runs ORDER BY id DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    # ================================================================ reset
    def reset_all(self):
        """Clear every table (used when loading a demo scenario)."""
        with self.connect() as conn:
            for table in ["transactions", "alerts", "budget_proposals", "recommendations",
                          "action_log", "analysis_runs"]:
                conn.execute(f"DELETE FROM {table}")
