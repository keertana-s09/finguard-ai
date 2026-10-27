"""FinGuard AI - REST API for the React frontend.

Place this file in the project root (next to app.py) and run:
    uvicorn api:app --reload --port 8000

It is a thin layer over the existing code: every endpoint calls FinGuardPipeline / FinanceDB,
so the React UI shows exactly what the Streamlit app shows.
"""
from __future__ import annotations

import io
import json
from datetime import date
from uuid import UUID
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field, field_validator

from src.config import DEFAULT_DB_PATH, EXPENSE_CATEGORIES, REPORTS_DIR
from src.pipeline import DEMO_SCENARIOS, FinGuardPipeline
from src.reports import markdown_report, transactions_csv

pipe: FinGuardPipeline | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipe
    pipe = FinGuardPipeline(DEFAULT_DB_PATH)  # loads (or trains) the ML models once
    yield


app = FastAPI(title="FinGuard AI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def records(df: pd.DataFrame) -> list:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null, numpy types become numbers)."""
    return [] if df is None or df.empty else json.loads(df.to_json(orient="records"))


def rerun() -> dict:
    result = pipe.run_analysis()
    return {"ok": bool(result.get("ok")), "message": result.get("message", "Analysis updated.")}


# ------------------------------------------------------------------ read endpoints
@app.get("/api/meta")
def meta():
    scenarios = {k: {"title": v["title"], "story": v["story"], "file": v["file"],
                     "savings_goal": v["settings"]["savings_goal"]} for k, v in DEMO_SCENARIOS.items()}
    return {"categories": list(EXPENSE_CATEGORIES), "scenarios": scenarios, "settings": pipe.get_settings()}


@app.get("/api/overview")
def overview():
    db = pipe.db
    tx = db.get_transactions()
    run = db.latest_run()
    monthly, cats = [], []
    if not tx.empty:
        t = tx.assign(month=pd.to_datetime(tx["date"]).dt.to_period("M").astype(str))
        g = t.pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0)
        monthly = [{"month": m, "income": float(r.get("Income", 0)), "expense": float(r.get("Expense", 0))}
                   for m, r in g.iterrows()]
        cur = t["month"].max()
        c = t[(t["type"] == "Expense") & (t["month"] == cur)].groupby("category")["amount"].sum()
        cats = [{"category": k, "amount": float(v)} for k, v in c.sort_values(ascending=False).items()]
    return {"n_transactions": int(len(tx)), "n_active_alerts": int(len(db.list_alerts("ACTIVE"))),
            "monthly_budget": pipe.monthly_budget_summary(),
            "run": run if run and run.get("ok") else None, "monthly": monthly, "categories": cats}


@app.get("/api/transactions")
def transactions(txn_type: str | None = Query(None, alias="type"), flagged: bool = False):
    tx = pipe.db.get_transactions()
    if txn_type:
        tx = tx[tx["type"] == txn_type]
    if flagged:
        tx = tx[tx["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False)
    return records(tx)


@app.get("/api/alerts")
def alerts(status: str | None = None):
    return records(pipe.db.list_alerts(status))


@app.get("/api/proposals")
def proposals():
    return {"latest": pipe.db.latest_proposal(), "all": records(pipe.db.list_proposals())}


@app.get("/api/recommendations")
def recommendations():
    return records(pipe.db.list_recommendations("ACTIVE"))


@app.get("/api/actions")
def actions():
    return records(pipe.db.list_actions())


@app.get("/api/metrics")
def metrics():
    path = REPORTS_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@app.get("/api/report")
def report():
    return {"markdown": markdown_report(pipe.db)}


@app.get("/api/report.md")
def report_download():
    return PlainTextResponse(markdown_report(pipe.db), media_type="text/markdown",
                             headers={"Content-Disposition": "attachment; filename=finguard_report.md"})


@app.get("/api/transactions.csv")
def transactions_download():
    return Response(transactions_csv(pipe.db), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=finguard_transactions.csv"})


# ------------------------------------------------------------------ actions
class ExpenseBody(BaseModel):
    date: date
    description: str = Field(min_length=1, max_length=500)
    amount: float = Field(gt=0, allow_inf_nan=False)
    category: str | None = None
    entry_id: UUID


@app.post("/api/expenses", status_code=201)
def add_expense(body: ExpenseBody):
    result = pipe.add_expense(body.date.isoformat(), body.description, body.amount,
                              body.category, str(body.entry_id))
    if not result["ok"]:
        raise HTTPException(422, "; ".join(result["errors"]))
    return result


@app.post("/api/scenarios/{key}")
def load_scenario(key: str):
    if key not in DEMO_SCENARIOS:
        raise HTTPException(404, f"Unknown scenario '{key}'")
    out = pipe.load_demo(key)
    return {"inserted": out["import"]["inserted"], "status": out["analysis"]["assessment"]["status"]}


@app.post("/api/upload")
def upload(file: UploadFile = File(...), replace: bool = Form(False)):
    if replace:
        pipe.db.reset_all()
    result = pipe.import_csv(io.BytesIO(file.file.read()))
    if result["ok"]:
        pipe.run_analysis()
    return result


@app.post("/api/analysis/run")
def run_analysis():
    return rerun()


@app.delete("/api/data")
def clear_data():
    pipe.db.reset_all()
    return {"ok": True}


class CategoryBody(BaseModel):
    category: str


def require_transaction(pk: int):
    if pipe.db.get_review_status(pk) is None:
        raise HTTPException(404, f"Transaction {pk} not found")


@app.post("/api/transactions/{pk}/category")
def correct_category(pk: int, body: CategoryBody):
    if body.category not in EXPENSE_CATEGORIES:
        raise HTTPException(400, "Unknown category")
    require_transaction(pk)
    pipe.db.correct_category(pk, body.category)
    return rerun()


class ReviewBody(BaseModel):
    reviewed: bool


@app.post("/api/transactions/{pk}/review")
def review(pk: int, body: ReviewBody):
    require_transaction(pk)
    if body.reviewed:
        pipe.db.set_review_status(pk, "REVIEWED", "Confirmed by user")
    else:
        pipe.db.set_review_status(pk, "NONE", "")
    return rerun()


class SettingsBody(BaseModel):
    # Same limits as the Streamlit form: amounts cannot be negative.
    monthly_income: float = Field(0.0, ge=0)
    savings_goal: float = Field(0.0, ge=0)
    monthly_budget: float = Field(0.0, ge=0)
    essential_categories: list[str]

    @field_validator("essential_categories")
    @classmethod
    def known_categories(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(EXPENSE_CATEGORIES))
        if unknown:
            raise ValueError(f"unknown categories: {', '.join(unknown)}")
        return value


@app.put("/api/settings")
def save_settings(body: SettingsBody):
    pipe.save_settings(**body.model_dump())
    return rerun()


@app.post("/api/alerts/{alert_key:path}/acknowledge")
def acknowledge(alert_key: str):
    alert = pipe.db.get_alert(alert_key)
    if alert is None:
        raise HTTPException(404, f"Alert '{alert_key}' not found")
    if alert["status"] != "ACTIVE":
        raise HTTPException(409, f"Only ACTIVE alerts can be acknowledged (this one is {alert['status']})")
    pipe.db.update_alert(alert_key, status="ACKNOWLEDGED")
    return {"ok": True}


@app.post("/api/proposals/{pid}/accept")
def accept_proposal(pid: int):
    if pid not in set(pipe.db.list_proposals()["id"]):
        raise HTTPException(404, f"Proposal {pid} not found")
    latest = pipe.db.latest_proposal()
    if latest is None or latest["id"] != pid or latest["status"] != "PROPOSED":
        raise HTTPException(409, "Only the latest PROPOSED budget proposal can be accepted")
    pipe.db.set_proposal_status(pid, "ACCEPTED")
    return {"ok": True}
