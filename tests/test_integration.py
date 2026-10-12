"""Test 13: end-to-end workflow and application startup."""
import io
import os


from src.config import PROJECT_ROOT
from src.financial_agent import ACHIEVABLE, INFEASIBLE, ON_TRACK
from tests.conftest import make_history


def test_demo_scenarios_produce_different_decisions(pipeline):
    statuses = {}
    for key in ["normal", "anomaly", "overspending", "infeasible"]:
        out = pipeline.load_demo(key)
        assert out["import"]["ok"] and out["analysis"]["ok"]
        statuses[key] = out["analysis"]["assessment"]["status"]
    assert statuses["normal"] == ON_TRACK
    assert statuses["overspending"] == ACHIEVABLE
    assert statuses["infeasible"] == INFEASIBLE


def test_anomaly_to_alert_to_dashboard_chain(pipeline):
    """Isolation Forest flags -> agent requests actions -> engine stores them -> review resolves them."""
    a = pipeline.load_demo("anomaly")["analysis"]
    big = {x["description"] for x in a["state"]["anomalies"]}
    assert "POS CROMA ELECTRONICS BANGALORE" in big
    alerts = pipeline.db.list_alerts("ACTIVE")
    unusual = alerts[alerts["alert_type"] == "UNUSUAL_TRANSACTION"]
    assert len(unusual) == 2
    tx = pipeline.db.get_transactions()
    croma = tx[tx["description"] == "POS CROMA ELECTRONICS BANGALORE"].iloc[0]
    assert croma["review_status"] == "PENDING_REVIEW"
    assert "typical" in croma["anomaly_explanation"]

    # user confirms the transaction -> next run resolves its alert automatically
    pipeline.db.set_review_status(int(croma["id"]), "REVIEWED")
    pipeline.run_analysis()
    alert = pipeline.db.get_alert(f"UNUSUAL_TRANSACTION:{croma['txn_id']}")
    assert alert["status"] == "RESOLVED"


def test_rerun_is_idempotent(pipeline):
    pipeline.load_demo("overspending")
    n_alerts = len(pipeline.db.list_alerts())
    n_props = len(pipeline.db.list_proposals())
    again = pipeline.run_analysis()
    assert len(pipeline.db.list_alerts()) == n_alerts
    assert len(pipeline.db.list_proposals()) == n_props
    assert all(r["outcome"] in ("UNCHANGED", "SKIPPED") for r in again["action_results"])


def test_settings_change_updates_decision_and_alerts(pipeline):
    pipeline.load_demo("overspending")
    assert pipeline.db.get_alert("FORECAST_OVERSPEND:2026-10")["status"] == "ACTIVE"
    pipeline.save_settings(savings_goal=1000.0)
    r = pipeline.run_analysis()
    assert r["assessment"]["status"] == ON_TRACK
    assert pipeline.db.get_alert("FORECAST_OVERSPEND:2026-10")["status"] == "RESOLVED"


def test_category_correction_survives_reanalysis(pipeline):
    pipeline.import_csv(io.StringIO(make_history(months=4).to_csv(index=False)))
    pipeline.run_analysis()
    tx = pipeline.db.get_transactions()
    pk = int(tx[tx["description"] == "Amazon Shopping"]["id"].iloc[0])
    pipeline.db.correct_category(pk, "Education")
    pipeline.run_analysis()
    row = pipeline.db.get_transactions().set_index("id").loc[pk]
    assert row["category"] == "Education" and row["category_source"] == "corrected"


def test_upload_flow_end_to_end(pipeline):
    res = pipeline.import_csv(PROJECT_ROOT / "data" / "sample_transactions.csv")
    assert res["ok"] and res["inserted"] == 22 and res["invalid_rows"] == 2 and res["duplicates_in_file"] == 1
    out = pipeline.run_analysis()
    assert out["ok"]
    assert out["forecast"]["method"] == "rule_average"          # only 2 months -> honest fallback
    assert pipeline.db.latest_run()["run_id"] == out["run_id"]


def test_streamlit_app_starts_and_all_pages_render(tmp_path):
    """Runs app.py headlessly with Streamlit's testing API on a temporary database."""
    from streamlit.testing.v1 import AppTest
    os.environ["FINGUARD_DB"] = str(tmp_path / "app.db")
    try:
        import importlib

        import src.config
        importlib.reload(src.config)
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=300)
        at.run()
        assert not at.exception
        at.sidebar.radio[0].set_value("📤 Upload Transactions").run()
        [b for b in at.button if b.key == "demo_overspending"][0].click().run()
        assert not at.exception
        for page in at.sidebar.radio[0].options:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, page
    finally:
        os.environ.pop("FINGUARD_DB", None)
        importlib.reload(src.config)
    assert (tmp_path / "app.db").exists()
