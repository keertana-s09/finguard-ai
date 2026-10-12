"""Tests 10-12: alert persistence, duplicate prevention, budget proposal saving."""
import pandas as pd
import pytest

from src.automated_response import AutomatedResponseEngine
from src.database import FinanceDB


@pytest.fixture
def db(tmp_path):
    return FinanceDB(tmp_path / "resp.db")


def alert(key="FORECAST_OVERSPEND:2026-10", message="over by 2000", severity="MEDIUM"):
    return {"action": "CREATE_ALERT", "alert_type": key.split(":")[0], "key": key, "severity": severity,
            "title": "Projected overspending", "message": message}


def proposal(total=23000.0, cut=2000.0):
    return {"action": "SAVE_BUDGET_PROPOSAL", "target_month": "2026-10", "feasibility": "ACHIEVABLE_WITH_CUTS",
            "income": 30000.0, "savings_goal": 7000.0, "spending_limit": 23000.0, "predicted_spending": 25000.0,
            "revised_total": total, "plan": {"reductions": [{"category": "Shopping", "reduction": cut}],
                                            "reason": "test", "forecast_method": "ml_ridge"}}


# 10. alert creation and persistence
def test_alert_is_created_and_persisted(db, tmp_path):
    res = AutomatedResponseEngine(db).execute([alert()], run_id="r1")
    assert res[0]["outcome"] == "CREATED"
    reopened = FinanceDB(tmp_path / "resp.db")         # new connection = survives restart
    stored = reopened.list_alerts()
    assert len(stored) == 1 and stored.iloc[0]["status"] == "ACTIVE"
    assert len(reopened.list_actions()) == 1


# 11. duplicate alert prevention
def test_duplicate_alert_is_not_created(db):
    engine = AutomatedResponseEngine(db)
    engine.execute([alert()], run_id="r1")
    res = engine.execute([alert()], run_id="r2")
    assert res[0]["outcome"] == "UNCHANGED"
    assert len(db.list_alerts()) == 1


def test_changed_alert_is_updated_not_duplicated(db):
    engine = AutomatedResponseEngine(db)
    engine.execute([alert()], run_id="r1")
    res = engine.execute([alert(message="over by 3000", severity="HIGH")], run_id="r2")
    assert res[0]["outcome"] == "UPDATED"
    stored = db.list_alerts()
    assert len(stored) == 1 and stored.iloc[0]["severity"] == "HIGH"


def test_alert_is_resolved_when_condition_disappears_and_reactivated(db):
    engine = AutomatedResponseEngine(db)
    engine.execute([alert()], run_id="r1")
    res = engine.execute([], run_id="r2")
    assert any(r["action"] == "RESOLVE_ALERT" for r in res)
    assert db.list_alerts().iloc[0]["status"] == "RESOLVED"
    engine.execute([alert()], run_id="r3")
    assert db.list_alerts().iloc[0]["status"] == "ACTIVE"


def test_acknowledged_alert_stays_acknowledged(db):
    engine = AutomatedResponseEngine(db)
    engine.execute([alert()], run_id="r1")
    db.update_alert("FORECAST_OVERSPEND:2026-10", status="ACKNOWLEDGED")
    engine.execute([alert()], run_id="r2")
    assert db.list_alerts().iloc[0]["status"] == "ACKNOWLEDGED"


# 12. budget proposal saving
def test_budget_proposal_is_saved_once_and_superseded_on_change(db):
    engine = AutomatedResponseEngine(db)
    assert engine.execute([proposal()], "r1")[0]["outcome"] == "CREATED"
    assert engine.execute([proposal()], "r2")[0]["outcome"] == "UNCHANGED"
    assert engine.execute([proposal(total=22000.0, cut=3000.0)], "r3")[0]["outcome"] == "CREATED"
    props = db.list_proposals()
    assert list(props["status"]) == ["PROPOSED", "SUPERSEDED"]
    latest = db.latest_proposal("2026-10")
    assert latest["revised_total"] == 22000.0 and latest["plan"]["reductions"][0]["reduction"] == 3000.0


def test_transaction_review_flag_lifecycle(db):
    db.insert_transactions(pd.DataFrame([{"txn_id": "A1", "date": "2026-09-01", "description": "Croma",
                                          "amount": 38000.0, "type": "Expense", "category": "Shopping"}]))
    pk = int(db.get_transactions()["id"].iloc[0])
    engine = AutomatedResponseEngine(db)
    flag = {"action": "FLAG_TRANSACTION", "txn_pk": pk, "reason": "unusual"}
    assert engine.execute([flag], "r1")[0]["outcome"] == "FLAGGED"
    assert db.get_review_status(pk) == "PENDING_REVIEW"
    db.set_review_status(pk, "REVIEWED")
    assert engine.execute([flag], "r2")[0]["outcome"] == "SKIPPED"
    db.set_review_status(pk, "PENDING_REVIEW")
    engine.execute([], "r3")                       # no longer flagged -> flag cleared automatically
    assert db.get_review_status(pk) == "NONE"


def test_recommendations_are_deduplicated_and_resolved(db):
    engine = AutomatedResponseEngine(db)
    rec = {"action": "CREATE_RECOMMENDATION", "key": "REDUCE:2026-10:Shopping", "kind": "REDUCE_CATEGORY",
           "message": "Reduce shopping", "amount": 2000}
    assert engine.execute([rec], "r1")[0]["outcome"] == "CREATED"
    assert engine.execute([rec], "r2")[0]["outcome"] == "UNCHANGED"
    engine.execute([], "r3")
    assert db.list_recommendations("ACTIVE").empty
