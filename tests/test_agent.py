"""Tests 7-9: achievable goals, infeasible goals, agent action generation."""
from src.financial_agent import (ACHIEVABLE, INFEASIBLE, ON_TRACK, AgentSettings, FinancialDecisionAgent,
                                 FinancialState)

BASELINE = {"Rent": 8000, "Food": 5000, "Shopping": 4000, "Entertainment": 2000, "Transport": 2000,
            "Bills": 2000, "Other": 1000}   # = 24,000


def make_agent(income=30000, predicted=25000, goal=7000, anomalies=None, essentials=None, budget=0.0):
    settings = AgentSettings(savings_goal=goal, monthly_budget=budget,
                             **({"essential_categories": essentials} if essentials else {}))
    agent = FinancialDecisionAgent(settings)
    agent.state = FinancialState(
        current_month="2026-09", income=income, income_source="test", current_spending=predicted,
        current_by_category=dict(BASELINE), baseline_by_category=dict(BASELINE), forecast_month="2026-10",
        predicted_spending=predicted, forecast_method="ml_ridge", anomalies=anomalies or [], flagged_txn_ids=[],
        savings_goal=goal, monthly_budget=budget, essential_categories=settings.essential_categories)
    return agent


def plan_for(agent):
    agent.evaluate_goals()
    return agent.generate_plan()


# 7. achievable savings goal (the README example: 30k income, 25k predicted, 7k goal)
def test_achievable_goal_closes_the_gap():
    agent = make_agent()
    plan = plan_for(agent)
    assert agent.assessment["status"] == ACHIEVABLE
    assert agent.assessment["spending_limit"] == 23000
    assert round(agent.assessment["gap"]) == 2000
    assert abs(sum(r["reduction"] for r in plan["reductions"]) - 2000) < 0.01
    assert plan["revised_total"] <= 23000 + 0.01
    assert plan["expected_savings"] >= 7000 - 0.01


def test_essential_categories_are_never_cut():
    agent = make_agent(essentials=["Rent", "Bills", "Food"])
    plan = plan_for(agent)
    cut = {r["category"] for r in plan["reductions"]}
    assert not cut & {"Rent", "Bills", "Food"}


def test_cuts_respect_maximum_fraction():
    plan = plan_for(make_agent(goal=9000))   # needs 4,000
    for r in plan["reductions"]:
        assert r["reduction"] <= r["planned"] * 0.5 + 0.01


def test_on_track_needs_no_cuts():
    agent = make_agent(predicted=20000, goal=7000)
    plan = plan_for(agent)
    assert agent.assessment["status"] == ON_TRACK
    assert plan["reductions"] == []


def test_budget_cap_tightens_the_limit():
    agent = make_agent(predicted=22000, goal=5000, budget=21000)
    plan_for(agent)
    assert agent.assessment["spending_limit"] == 21000
    assert agent.assessment["status"] == ACHIEVABLE


# 8. infeasible savings goal
def test_infeasible_goal_is_detected_not_faked():
    agent = make_agent(goal=15000)
    plan = plan_for(agent)
    a = agent.assessment
    assert a["status"] == INFEASIBLE
    assert plan["expected_savings"] < 15000                      # no impossible promise
    assert abs(plan["expected_savings"] - a["max_achievable_savings"]) < 1  # rupee rounding
    kinds = {x.get("kind") for x in plan["actions"]}
    assert "REALISTIC_GOAL" in kinds


def test_zero_income_is_infeasible():
    agent = make_agent(income=0, goal=1000)
    plan_for(agent)
    assert agent.assessment["status"] == INFEASIBLE


# 9. agent action generation
def _types(plan):
    return [(a["action"], a.get("alert_type")) for a in plan["actions"]]


def test_actions_for_achievable_case():
    plan = plan_for(make_agent())
    t = _types(plan)
    assert ("CREATE_ALERT", "FORECAST_OVERSPEND") in t
    assert ("SAVE_BUDGET_PROPOSAL", None) in t
    assert any(a["action"] == "CREATE_RECOMMENDATION" and a["kind"] == "REDUCE_CATEGORY" for a in plan["actions"])


def test_anomaly_influences_decision():
    anomaly = {"id": 7, "txn_id": "T7", "date": "2026-09-20", "month": "2026-09", "description": "Croma",
               "category": "Transport", "amount": 9000.0, "typical_amount": 300.0, "ratio": 30.0,
               "anomaly_score": 0.12, "explanation": "x", "review_status": "NONE"}
    plain = plan_for(make_agent())
    with_anomaly_agent = make_agent(anomalies=[anomaly])
    with_anomaly = plan_for(with_anomaly_agent)
    # the category with the anomaly is now cut first
    assert plain["reductions"][0]["category"] != "Transport"
    assert with_anomaly["reductions"][0]["category"] == "Transport"
    assert "unusually high transaction" in with_anomaly["reductions"][0]["reason"]
    # review flag + high-severity alert are requested
    t = _types(with_anomaly)
    assert ("FLAG_TRANSACTION", None) in t
    alert = [a for a in with_anomaly["actions"] if a.get("alert_type") == "UNUSUAL_TRANSACTION"][0]
    assert alert["severity"] == "HIGH"       # 9,000 >= 10% of 30,000 income


def test_reviewed_anomaly_creates_no_new_flag():
    anomaly = {"id": 7, "txn_id": "T7", "date": "2026-09-20", "month": "2026-09", "description": "Croma",
               "category": "Shopping", "amount": 9000.0, "typical_amount": 300.0, "ratio": 30.0,
               "anomaly_score": 0.12, "explanation": "x", "review_status": "REVIEWED"}
    plan = plan_for(make_agent(anomalies=[anomaly]))
    assert not any(a["action"] == "FLAG_TRANSACTION" for a in plan["actions"])
    assert not any(a.get("alert_type") == "UNUSUAL_TRANSACTION" for a in plan["actions"])
