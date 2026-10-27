"""FinGuard AI - Streamlit dashboard.

Run from the project root:
    streamlit run app.py
"""
import json
import uuid
from datetime import date

import pandas as pd
import streamlit as st

from src import charts
from src.config import DATA_DIR, DEFAULT_DB_PATH, EXPENSE_CATEGORIES, REPORTS_DIR
from src.pipeline import DEMO_SCENARIOS, FinGuardPipeline
from src.daily_budget import format_daily_amount
from src.reports import markdown_report, transactions_csv
from src.utils import format_inr

st.set_page_config(page_title="FinGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.fg-card {background:#ffffff; border:1px solid rgba(11,11,11,0.10); border-radius:12px; padding:14px 16px;
          height:100%; min-height:118px;}
.fg-card .label {color:#52514e; font-size:0.82rem; margin-bottom:4px;}
.fg-card .value {color:#0b0b0b; font-size:1.55rem; font-weight:650; line-height:1.2;}
.fg-card .sub {color:#898781; font-size:0.78rem; margin-top:4px;}
.fg-badge {display:inline-block; padding:3px 10px; border-radius:999px; font-weight:600; font-size:0.85rem;}
.fg-ok {background:#e3f5e3; color:#006300;}
.fg-warn {background:#fff1d6; color:#7a4f00;}
.fg-bad {background:#fbe3e3; color:#9b1c1c;}
.fg-step {border-left:3px solid #2a78d6; padding:2px 0 2px 12px; margin:6px 0 14px 0;}
</style>
""", unsafe_allow_html=True)


# ====================================================================== shared state
@st.cache_resource(show_spinner="Loading ML models (first run trains them, ~1 minute)...")
def get_pipeline() -> FinGuardPipeline:
    return FinGuardPipeline(DEFAULT_DB_PATH)


pipe = get_pipeline()
db = pipe.db


def run_analysis(message: str = "Analysis updated."):
    with st.spinner("Running classifier → Isolation Forest → forecast → agent → automated response..."):
        result = pipe.run_analysis()
    st.session_state["flash"] = message if result.get("ok") else result.get("message")


def card(col, label, value, sub=""):
    col.markdown(f'<div class="fg-card"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


STATUS_BADGE = {"ON_TRACK": ("On track", "fg-ok"), "ACHIEVABLE_WITH_CUTS": ("Achievable with cuts", "fg-warn"),
                "INFEASIBLE": ("Goal infeasible", "fg-bad")}
SEVERITY_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🔵"}


def status_badge(status):
    text, cls = STATUS_BADGE.get(status, (status, "fg-warn"))
    return f'<span class="fg-badge {cls}">{text}</span>'


def need_data() -> bool:
    if db.count_transactions() == 0 or not (db.latest_run() or {}).get("ok"):
        st.info("No analysed data yet. Add an expense on Home Dashboard, or import past transactions on Upload Transactions.")
        return True
    return False


def load_eval_metrics():
    path = REPORTS_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def daily_expenses():
    st.subheader("Add a daily expense")
    epoch = st.session_state.get("expense_form_epoch", 0)
    token = st.session_state.setdefault("expense_entry_id", str(uuid.uuid4()))
    with st.form(f"daily_expense_{epoch}"):
        c = st.columns(2)
        day = c[0].date_input("Expense date", value=date.today(), max_value=date.today())
        amount = c[1].number_input("Expense amount (₹)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
        description = st.text_input("Expense description", max_chars=500, placeholder="e.g. Lunch at college")
        category = st.selectbox("Expense category", ["Predict automatically"] + list(EXPENSE_CATEGORIES))
        if st.form_submit_button("Save expense", type="primary"):
            with st.spinner("Saving expense and updating analysis..."):
                result = pipe.add_expense(day.isoformat(), description, amount,
                                           None if category == "Predict automatically" else category, token)
            if not result["ok"]:
                for error in result["errors"]:
                    st.error(error)
            else:
                st.session_state["expense_entry_id"] = str(uuid.uuid4())
                st.session_state["expense_form_epoch"] = epoch + 1
                st.session_state["flash"] = "Expense saved. Your remaining monthly budget is updated."
                if not result["analysis_ok"]:
                    st.session_state["expense_analysis_warning"] = True
                st.rerun()
    if st.session_state.pop("expense_analysis_warning", False):
        st.warning("Your expense was saved, but analysis could not refresh. Use Re-run AI analysis; do not enter it again.")
    daily = pipe.monthly_budget_summary()
    st.subheader(f"Remaining monthly budget ({daily['month']})")
    c = st.columns(3)
    c[0].metric("Monthly spending limit", format_daily_amount(daily["spending_limit"]) if daily["spending_limit"] is not None else "Not set")
    c[1].metric("Recorded expenses this month", format_daily_amount(daily["spent"]))
    c[2].metric("Remaining budget", format_daily_amount(daily["remaining"]) if daily["remaining"] is not None else "Not set")
    if daily["remaining"] is not None and daily["remaining"] < 0:
        st.warning(daily["recommendation"])
    else:
        st.info(daily["recommendation"])
    for suggestion in daily["suggestions"]:
        st.write(suggestion)


# ====================================================================== sidebar
PAGES = ["🏠 Home Dashboard", "📤 Upload Transactions", "🏷️ Expense Classification", "🚨 Anomaly Detection",
         "📈 Spending Prediction", "🤖 AI Financial Advisor", "🔔 Alerts and Reports"]
with st.sidebar:
    st.markdown("## 🛡️ FinGuard AI")
    st.caption("Intelligent personal finance & budget optimisation agent")
    try:  # optional deep link, e.g. http://localhost:8501/?page=4
        start_page = min(max(int(st.query_params.get("page", 1)) - 1, 0), len(PAGES) - 1)
    except ValueError:
        start_page = 0
    page = st.radio("Navigation", PAGES, index=start_page, label_visibility="collapsed")
    st.divider()
    run = db.latest_run()
    n_active = len(db.list_alerts("ACTIVE"))
    st.markdown(f"**Transactions:** {db.count_transactions()}")
    st.markdown(f"**Active alerts:** {n_active}")
    if run and run.get("ok"):
        st.markdown("**Agent status:** " + status_badge(run["assessment"]["status"]), unsafe_allow_html=True)
        st.caption(f"Last analysis: {run['created_at']}")
    if st.button("🔄 Re-run AI analysis", width="stretch", disabled=db.count_transactions() == 0):
        run_analysis()
        st.rerun()
    st.divider()
    st.caption("Components: Isolation Forest anomaly detection → goal-based agent → automated response. "
               "Supporting ML: TF-IDF + Logistic Regression, Ridge Regression.")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

run = db.latest_run()

# ====================================================================== 1. HOME
if page == PAGES[0]:
    st.title("Home Dashboard")
    daily_expenses()
    if not need_data():
        tx = db.get_transactions()
        s, a, plan, fc = run["state"], run["assessment"], run["plan"], run["forecast"]
        st.markdown(f"Latest month in data: **{s['current_month']}** · Forecast month: **{fc['target_month']}** · "
                    f"Agent decision: {status_badge(a['status'])}", unsafe_allow_html=True)
        c = st.columns(6)
        card(c[0], "Monthly income", format_inr(s["income"]), s["income_source"][:40])
        card(c[1], f"Expenses ({s['current_month']})", format_inr(s["current_spending"]))
        card(c[2], "Remaining balance", format_inr(s["income"] - s["current_spending"]),
             "income − this month's expenses")
        card(c[3], "Predicted next month", format_inr(fc["predicted"] or 0),
             "ML (Ridge)" if fc["method"] == "ml_ridge" else "rule-based fallback")
        card(c[4], "Savings goal", format_inr(s["savings_goal"]), f"limit {format_inr(a['spending_limit'])}")
        card(c[5], "Unusual transactions", str(run["n_flagged"]),
             f"{len(s['anomalies'])} unusually high")
        st.write("")
        left, right = st.columns([1.15, 1])
        left.plotly_chart(charts.monthly_trend(tx))
        right.plotly_chart(charts.category_bar(tx, s["current_month"]))

        left, right = st.columns([1.15, 1])
        left.plotly_chart(charts.forecast_chart(fc))
        with right:
            st.markdown("**Savings goal progress (next month, with recommended budget)**")
            st.plotly_chart(charts.savings_gauge(plan["expected_savings"], s["savings_goal"]))
            st.caption(f"Bar = expected savings {format_inr(plan['expected_savings'])}; black line = goal "
                       f"{format_inr(s['savings_goal'])}.")
            alerts = db.list_alerts("ACTIVE")
            st.markdown(f"**Active alerts ({len(alerts)})**")
            for r in alerts.head(4).itertuples():
                st.markdown(f"{SEVERITY_ICON.get(r.severity, '')} **{r.severity}** · {r.title}")
            if alerts.empty:
                st.caption("No active alerts.")
        st.plotly_chart(charts.budget_comparison(plan["revised_budget"]))
        anomalies = pd.DataFrame(s["anomalies"])
        if not anomalies.empty:
            # review status may have changed after the run (response engine / user) -> read it live
            anomalies["review_status"] = anomalies["id"].map(tx.set_index("id")["review_status"])
            st.markdown("**Unusually high transactions**")
            st.dataframe(anomalies[["date", "description", "category", "amount", "typical_amount",
                                    "anomaly_score", "review_status"]], hide_index=True)

# ====================================================================== 2. UPLOAD
elif page == PAGES[1]:
    st.title("Upload Transactions")
    daily_expenses()
    st.markdown("#### Option A - Demonstration scenarios")
    st.caption("Each scenario resets the database, imports its CSV and runs the full pipeline. "
               "Only the input file and user settings are predefined; every output is computed live.")
    cols = st.columns(len(DEMO_SCENARIOS))
    for col, (key, sc) in zip(cols, DEMO_SCENARIOS.items()):
        with col.container(border=True):
            st.markdown(f"**{sc['title']}**")
            st.caption(sc["story"])
            st.caption(f"File: `data/{sc['file']}` · goal {format_inr(sc['settings']['savings_goal'])}")
            if st.button("Load scenario", key=f"demo_{key}", type="primary", width="stretch"):
                with st.spinner("Loading scenario and running the AI pipeline..."):
                    out = pipe.load_demo(key)
                st.session_state["flash"] = (f"Loaded '{sc['title']}': {out['import']['inserted']} transactions. "
                                             f"Agent decision: {out['analysis']['assessment']['status']}.")
                st.rerun()

    st.markdown("#### Option B - Upload your own CSV")
    st.markdown("Required columns: **Date, Description, Amount, Type** (Income/Expense). Optional: "
                "**Transaction ID, Category** (blank categories are predicted by the ML model). Dates may be "
                "`YYYY-MM-DD` or `DD/MM/YYYY`; amounts may contain ₹ and commas.")
    c1, c2 = st.columns([2, 1])
    replace = c2.checkbox("Clear existing data before import", value=False)
    sample_path = DATA_DIR / "sample_transactions.csv"
    c2.download_button("Download sample CSV", sample_path.read_bytes(), "sample_transactions.csv", "text/csv")
    uploaded = c1.file_uploader("Transaction CSV", type=["csv"])
    if uploaded is not None and st.button("Validate and import", type="primary"):
        if replace:
            db.reset_all()
        result = pipe.import_csv(uploaded)
        if not result["ok"]:
            for e in result["errors"]:
                st.error(e)
        else:
            st.success(f"Imported {result['inserted']} new transaction(s) out of {result['n_input']} rows.")
            run_analysis(f"Imported {result['inserted']} transaction(s) and ran the AI pipeline.")
        for w in result["warnings"]:
            st.warning(w)
        if result["ok"]:
            st.rerun()

    st.markdown("#### Current data")
    tx = db.get_transactions()
    if tx.empty:
        st.caption("Database is empty.")
    else:
        st.dataframe(tx[["txn_id", "date", "description", "amount", "type", "category", "category_source"]],
                     hide_index=True, height=320)
        if st.button("🗑️ Clear all data"):
            db.reset_all()
            st.session_state["flash"] = "All data cleared."
            st.rerun()

# ====================================================================== 3. CLASSIFICATION
elif page == PAGES[2]:
    st.title("Expense Classification")
    st.markdown('<div class="fg-step">TF-IDF (word + character n-grams) → Logistic Regression. '
                'Confidence = highest class probability; below 0.60 the transaction is marked for review. '
                'Categories you entered or corrected are never overwritten.</div>', unsafe_allow_html=True)
    if not need_data():
        tx = db.get_transactions()
        exp = tx[tx["type"] == "Expense"].copy()
        c = st.columns(4)
        card(c[0], "Expenses", str(len(exp)))
        card(c[1], "Predicted by ML", str(int((exp["category_source"] == "model").sum())))
        card(c[2], "Low confidence (review)", str(int(exp["needs_review"].sum())))
        card(c[3], "Corrected by you", str(int((exp["category_source"] == "corrected").sum())))
        st.write("")
        only_review = st.toggle("Show only low-confidence predictions", value=False)
        view = exp[exp["needs_review"] == 1] if only_review else exp
        st.dataframe(view[["id", "date", "description", "amount", "category", "category_source",
                           "predicted_category", "category_confidence", "needs_review"]],
                     hide_index=True, height=360,
                     column_config={"category_confidence": st.column_config.ProgressColumn(
                         "confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                         "needs_review": st.column_config.CheckboxColumn("review?")})

        st.markdown("#### Correct a category")
        options = exp.sort_values(["needs_review", "category_confidence"], ascending=[False, True])
        labels = {int(r.id): f"#{r.id} · {r.date} · {r.description} · {format_inr(r.amount)} · now: {r.category}"
                  for r in options.itertuples()}
        with st.form("correct"):
            txn = st.selectbox("Transaction", list(labels), format_func=labels.get)
            new_cat = st.selectbox("Correct category", EXPENSE_CATEGORIES)
            if st.form_submit_button("Save correction and re-run analysis", type="primary"):
                db.correct_category(txn, new_cat)
                run_analysis(f"Transaction #{txn} set to {new_cat}; anomalies, forecast and agent updated.")
                st.rerun()

        metrics = load_eval_metrics()
        with st.expander("Model evaluation (held-out test set, synthetic data)", expanded=False):
            if metrics:
                cm = metrics["classification"]
                c = st.columns(4)
                card(c[0], "ML accuracy", f"{cm['ml']['accuracy']:.1%}")
                card(c[1], "ML macro F1", f"{cm['ml']['macro_f1']:.3f}")
                card(c[2], "Keyword baseline accuracy", f"{cm['keyword_baseline']['accuracy']:.1%}")
                card(c[3], "Keyword baseline macro F1", f"{cm['keyword_baseline']['macro_f1']:.3f}")
                st.caption(f"Test merchants never seen in training ({cm['unseen_test_merchants']} merchants).")
                st.plotly_chart(charts.confusion_heatmap(cm["confusion_matrix"]))
            else:
                st.caption("Run `python scripts/evaluate_models.py` to generate evaluation metrics.")

# ====================================================================== 4. ANOMALY
elif page == PAGES[3]:
    st.title("Anomaly Detection")
    st.markdown('<div class="fg-step"><b>Isolation Forest</b> on expense transactions only. Features: log amount, '
                'excess over the typical amount for that category (from your previous transactions), day of week. '
                'Score: higher = more unusual. The written explanation is computed from category statistics - '
                'Isolation Forest itself only gives the score.</div>', unsafe_allow_html=True)
    if not need_data():
        st.caption(run["anomaly_message"])
        tx = db.get_transactions()
        scored = tx[(tx["type"] == "Expense") & tx["anomaly_score"].notna()]
        if scored.empty:
            st.warning("Not enough expense history for anomaly detection.")
        else:
            st.plotly_chart(charts.anomaly_scatter(scored))
            flagged = scored[scored["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False)
            st.markdown(f"#### Flagged transactions ({len(flagged)})")
            if flagged.empty:
                st.success("No unusual transactions found.")
            for r in flagged.itertuples():
                with st.container(border=True):
                    a, b, c = st.columns([3, 1.2, 1.3])
                    a.markdown(f"**{r.description}** · {r.date} · {r.category}")
                    a.markdown(f"_{r.anomaly_explanation}_")
                    b.metric("Amount", format_inr(r.amount))
                    b.caption(f"typical {format_inr(r.typical_amount)} · score {r.anomaly_score:.3f}")
                    c.markdown(f"Review status: **{r.review_status}**")
                    if r.review_status != "REVIEWED":
                        if c.button("✔ Confirm as legitimate", key=f"rev_{r.id}"):
                            db.set_review_status(r.id, "REVIEWED", "Confirmed by user")
                            run_analysis("Marked as reviewed - related alerts were resolved automatically.")
                            st.rerun()
                    else:
                        if c.button("↺ Re-open review", key=f"reopen_{r.id}"):
                            db.set_review_status(r.id, "NONE", "")
                            run_analysis("Review re-opened.")
                            st.rerun()
        metrics = load_eval_metrics()
        if metrics:
            with st.expander("Model evaluation (held-out months, synthetic labelled anomalies)"):
                an = metrics["anomaly"]
                rows = [{"method": k.replace("_", " "), **an[k]} for k in
                        ["isolation_forest", "isolation_forest_with_category_onehot", "rule_baseline_3x_typical"]]
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                st.caption(f"Contamination {an['contamination_selected_on_validation']} chosen on validation months.")

# ====================================================================== 5. PREDICTION
elif page == PAGES[4]:
    st.title("Spending Prediction")
    if not need_data():
        fc = run["forecast"]
        method = {"ml_ridge": "Machine learning (Ridge Regression)",
                  "rule_average": "RULE-BASED fallback (not ML)", "none": "No forecast"}[fc["method"]]
        c = st.columns(3)
        card(c[0], f"Predicted spending for {fc['target_month']}", format_inr(fc["predicted"] or 0))
        card(c[1], "Method", method)
        hist = pd.DataFrame(fc["history"])
        bt = hist.dropna(subset=["predicted"]) if not hist.empty else hist
        mae = (bt["predicted"] - bt["regular"]).abs().mean() if not bt.empty else None
        card(c[2], "Backtest MAE on your data", format_inr(mae) if mae is not None else "n/a",
             f"{len(bt)} month(s) with 3 months of history")
        st.info(fc["note"])
        st.plotly_chart(charts.forecast_chart(fc))
        st.caption("Regular spending = all expenses minus transactions flagged as unusually high (one-offs). "
                   "Backtest points use only the three months before each month - never future data.")
        if not hist.empty:
            st.dataframe(hist, hide_index=True)
        metrics = load_eval_metrics()
        if metrics:
            with st.expander("Model evaluation (time-based split, synthetic data)"):
                f = metrics["forecast"]
                st.dataframe(pd.DataFrame([
                    {"method": "Ridge Regression", **f["ridge"]},
                    {"method": "Historical average baseline", **f["historical_mean_baseline"]},
                    {"method": "3-month moving average baseline", **f["moving_average_3_baseline"]}]),
                    hide_index=True)
                st.caption(f"Test period {f['test_period']}, {f['test_points']} profile-months.")

# ====================================================================== 6. ADVISOR (AGENT)
elif page == PAGES[5]:
    st.title("AI Financial Advisor")
    st.caption("Goal-based agent: perceive → evaluate goals → generate plan → execute actions. "
               "All recommendations are simulated plans - no money is moved.")
    settings = pipe.get_settings()
    with st.form("settings"):
        st.markdown("#### Your goals and constraints")
        c = st.columns(3)
        income = c[0].number_input("Monthly income (₹, 0 = detect from data)", min_value=0.0, step=1000.0,
                                   value=float(settings["monthly_income"]))
        goal = c[1].number_input("Savings goal per month (₹)", min_value=0.0, step=500.0,
                                 value=float(settings["savings_goal"]))
        budget = c[2].number_input("Monthly spending cap (₹, 0 = none)", min_value=0.0, step=1000.0,
                                   value=float(settings["monthly_budget"]))
        essentials = st.multiselect("Essential categories (the agent will never cut these)", EXPENSE_CATEGORIES,
                                    default=settings["essential_categories"])
        if st.form_submit_button("Save and run the agent", type="primary"):
            pipe.save_settings(monthly_income=income, savings_goal=goal, monthly_budget=budget,
                               essential_categories=essentials)
            run_analysis("Settings saved - the agent re-planned and the response engine updated alerts.")
            st.rerun()

    if not need_data():
        s, a, plan = run["state"], run["assessment"], run["plan"]
        st.markdown("#### 1 · Perceive - current financial state")
        c = st.columns(5)
        card(c[0], "Income", format_inr(s["income"]), s["income_source"][:38])
        card(c[1], f"Spent in {s['current_month']}", format_inr(s["current_spending"]))
        card(c[2], f"Predicted {s['forecast_month']}", format_inr(s["predicted_spending"]),
             {"ml_ridge": "ML (Ridge)", "rule_average": "rule-based fallback"}.get(s["forecast_method"], "-"))
        card(c[3], "Savings goal", format_inr(s["savings_goal"]))
        card(c[4], "Unusually high txns", str(len(s["anomalies"])), "from Isolation Forest")

        st.markdown("#### 2 · Evaluate goals")
        st.markdown(f"{status_badge(a['status'])} &nbsp; {a['reason']}", unsafe_allow_html=True)
        c = st.columns(4)
        card(c[0], "Spending limit", format_inr(a["spending_limit"]), a["limit_reason"][:48])
        card(c[1], "Required reduction", format_inr(max(0, a["gap"])))
        card(c[2], "Max possible reduction", format_inr(a["max_possible_reduction"]), "flexible categories only")
        card(c[3], "Max achievable savings", format_inr(a["max_achievable_savings"]))
        st.write("")
        opts = pd.DataFrame(a["options"])
        if not opts.empty:
            st.markdown("**Options evaluated** (planned spending per category = forecast × recent share)")
            st.dataframe(opts.assign(max_cut_fraction=opts["max_cut_fraction"].map(lambda v: f"{v:.0%}")),
                         hide_index=True)

        st.markdown("#### 3 · Generate plan")
        if plan["reductions"]:
            for r in plan["reductions"]:
                st.markdown(f"- **{r['category']}**: {format_inr(r['planned'])} → **{format_inr(r['revised'])}** "
                            f"(−{format_inr(r['reduction'])}). _{r['reason']}_")
        else:
            st.markdown("- No reductions needed.")
        left, right = st.columns([1.4, 1])
        left.plotly_chart(charts.budget_comparison(plan["revised_budget"]))
        with right:
            st.markdown("**Savings with the recommended budget**")
            st.plotly_chart(charts.savings_gauge(plan["expected_savings"], s["savings_goal"]))
            st.caption(f"Revised total {format_inr(plan['revised_total'])}; expected savings "
                       f"{format_inr(plan['expected_savings'])} vs goal {format_inr(s['savings_goal'])}.")

        st.markdown("#### 4 · Execute actions (automated response, this run)")
        res = pd.DataFrame(run["action_results"])
        if res.empty:
            st.caption("No actions.")
        else:
            st.dataframe(res, hide_index=True)
            st.caption("CREATED/FLAGGED = new record written to SQLite · UNCHANGED = duplicate prevented · "
                       "RESOLVED = condition disappeared, so the record was closed automatically.")

# ====================================================================== 7. ALERTS & REPORTS
elif page == PAGES[6]:
    st.title("Alerts and Reports")
    if not need_data():
        alerts = db.list_alerts()
        tabs = st.tabs(["🔔 Alerts", "📋 Budget proposals", "💡 Recommendations", "⚙️ Action log", "📄 Export"])
        with tabs[0]:
            open_alerts = alerts[alerts["status"] != "RESOLVED"]
            if open_alerts.empty:
                st.success("No open alerts.")
            for r in open_alerts.itertuples():
                with st.container(border=True):
                    a, b = st.columns([5, 1])
                    a.markdown(f"{SEVERITY_ICON.get(r.severity, '')} **{r.severity}** · **{r.title}** "
                               f"· `{r.status}`")
                    a.write(r.message)
                    a.caption(f"{r.alert_type} · created {r.created_at} · updated {r.updated_at}")
                    if r.status == "ACTIVE" and b.button("Acknowledge", key=f"ack_{r.id}"):
                        db.update_alert(r.alert_key, status="ACKNOWLEDGED")
                        st.rerun()
            resolved = alerts[alerts["status"] == "RESOLVED"]
            if not resolved.empty:
                with st.expander(f"Resolved alerts ({len(resolved)})"):
                    st.dataframe(resolved[["severity", "title", "resolved_reason", "updated_at"]], hide_index=True)
        with tabs[1]:
            props = db.list_proposals()
            latest = db.latest_proposal()
            if latest:
                st.markdown(f"**Latest proposal #{latest['id']}** for {latest['target_month']} · "
                            f"{latest['feasibility']} · status `{latest['status']}`")
                st.markdown(f"Spending limit {format_inr(latest['spending_limit'])}, predicted "
                            f"{format_inr(latest['predicted_spending'])}, revised total "
                            f"{format_inr(latest['revised_total'])}.")
                if latest["status"] == "PROPOSED" and st.button("Accept this budget proposal", type="primary"):
                    db.set_proposal_status(latest["id"], "ACCEPTED")
                    st.rerun()
            st.dataframe(props, hide_index=True)
        with tabs[2]:
            recs = db.list_recommendations("ACTIVE")
            for r in recs.itertuples():
                st.markdown(f"- {r.message}")
            if recs.empty:
                st.caption("No active recommendations.")
        with tabs[3]:
            st.dataframe(db.list_actions(), hide_index=True, height=420)
        with tabs[4]:
            report = markdown_report(db)
            c1, c2 = st.columns(2)
            c1.download_button("⬇️ Download report (Markdown)", report, "finguard_report.md", "text/markdown")
            c2.download_button("⬇️ Download transactions (CSV)", transactions_csv(db),
                               "finguard_transactions.csv", "text/csv")
            st.markdown(report)
