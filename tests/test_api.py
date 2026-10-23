"""Tests for the FastAPI layer used by the React frontend (api.py).

Skipped automatically when the optional API packages are not installed
(pip install -r requirements-api.txt).
"""
import pytest

pytest.importorskip("fastapi")
# The test client needs an HTTP client package (httpx2 on current Starlette, httpx on older ones).
TestClient = pytest.importorskip("fastapi.testclient").TestClient

import api  # noqa: E402


@pytest.fixture
def client(pipeline, monkeypatch):
    """API client backed by the temporary-database pipeline (lifespan is not run)."""
    monkeypatch.setattr(api, "pipe", pipeline)
    return TestClient(api.app)


def test_overview_after_every_scenario_is_valid_json(client):
    for key in ["normal", "anomaly", "overspending", "infeasible"]:
        assert client.post(f"/api/scenarios/{key}").status_code == 200
        r = client.get("/api/overview")
        assert r.status_code == 200, key
        assert r.json()["run"]["forecast"]["history"][0]["predicted"] is None


def test_unknown_scenario_is_404(client):
    assert client.post("/api/scenarios/nope").status_code == 404


def test_settings_reject_negative_amounts_and_unknown_categories(client):
    client.post("/api/scenarios/normal")
    bad_goal = {"savings_goal": -5000, "essential_categories": ["Rent"]}
    assert client.put("/api/settings", json=bad_goal).status_code == 422
    bad_cat = {"savings_goal": 1000, "essential_categories": ["Rent", "Crypto"]}
    r = client.put("/api/settings", json=bad_cat)
    assert r.status_code == 422 and "Crypto" in r.text
    assert client.get("/api/meta").json()["settings"]["savings_goal"] == 10000   # unchanged
    ok = {"savings_goal": 12000, "essential_categories": ["Rent", "Bills"]}
    assert client.put("/api/settings", json=ok).status_code == 200


def test_only_latest_proposed_budget_can_be_accepted(client):
    client.post("/api/scenarios/overspending")
    client.put("/api/settings", json={"monthly_income": 30000, "savings_goal": 15000,
                                      "essential_categories": ["Rent", "Bills", "Education", "Healthcare"]})
    props = client.get("/api/proposals").json()
    statuses = {p["id"]: p["status"] for p in props["all"]}
    old = next(pid for pid, s in statuses.items() if s == "SUPERSEDED")
    assert client.post("/api/proposals/999999/accept").status_code == 404
    assert client.post(f"/api/proposals/{old}/accept").status_code == 409
    assert client.post(f"/api/proposals/{props['latest']['id']}/accept").status_code == 200
    assert client.get("/api/proposals").json()["latest"]["status"] == "ACCEPTED"


def test_acknowledge_checks_alert_exists_and_is_active(client):
    client.post("/api/scenarios/overspending")
    key = client.get("/api/alerts?status=ACTIVE").json()[0]["alert_key"]
    assert client.post("/api/alerts/NO_SUCH:2026-10/acknowledge").status_code == 404
    assert client.post(f"/api/alerts/{key}/acknowledge").status_code == 200
    assert client.post(f"/api/alerts/{key}/acknowledge").status_code == 409   # already acknowledged


def test_transaction_endpoints_404_for_unknown_id(client):
    client.post("/api/scenarios/normal")
    assert client.post("/api/transactions/999999/review", json={"reviewed": True}).status_code == 404
    assert client.post("/api/transactions/999999/category", json={"category": "Food"}).status_code == 404


def test_confirming_anomaly_resolves_its_alert(client):
    client.post("/api/scenarios/anomaly")
    flagged = client.get("/api/transactions?flagged=true").json()
    croma = next(t for t in flagged if "CROMA" in t["description"])
    assert croma["review_status"] == "PENDING_REVIEW"
    assert client.post(f"/api/transactions/{croma['id']}/review", json={"reviewed": True}).status_code == 200
    alerts = {a["alert_key"]: a["status"] for a in client.get("/api/alerts").json()}
    assert alerts[f"UNUSUAL_TRANSACTION:{croma['txn_id']}"] == "RESOLVED"
