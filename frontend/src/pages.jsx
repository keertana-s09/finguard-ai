import { useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Badge, Card, NeedData, Table, inr, request, send, useGet } from './ui.jsx';
import DailyExpenses from './DailyExpenses.jsx';

const post = (p, body) => send('POST', p, body);
const axis = { stroke: 'var(--mute)', fontSize: 12 };
const Chart = ({ children, data, kind: K = BarChart }) => (
  <div className="panel"><ResponsiveContainer width="100%" height={240}>
    <K data={data}><CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />{children}<Tooltip formatter={inr} /><Legend /></K>
  </ResponsiveContainer></div>
);

function ForecastChart({ fc }) {
  const data = [...(fc.history || []).map((h) => ({ m: h.month ?? h.target_month ?? '', regular: h.regular, predicted: h.predicted })),
                { m: fc.target_month, predicted: fc.predicted }];
  return (
    <Chart data={data} kind={LineChart}>
      <XAxis dataKey="m" {...axis} /><YAxis {...axis} />
      <Line dataKey="regular" name="Regular spending" stroke="var(--brand)" strokeWidth={2} />
      <Line dataKey="predicted" name="Predicted" stroke="var(--accent)" strokeWidth={2} strokeDasharray="5 4" />
    </Chart>
  );
}

// ------------------------------------------------------------------ 1. Home
export function Home({ ov, tick, act, busy }) {
  const [alerts] = useGet('/alerts?status=ACTIVE', tick);
  // The saved run is a snapshot taken BEFORE the response engine flags transactions,
  // so read the current review status from the database instead.
  const [flagged] = useGet('/transactions?flagged=true', tick);
  if (!ov.run) return <><h2>Home Dashboard</h2><DailyExpenses ov={ov} act={act} busy={busy} /><NeedData /></>;
  const { state: s, assessment: a, plan, forecast: fc } = ov.run;
  const liveStatus = Object.fromEntries((flagged || []).map((t) => [t.id, t.review_status]));
  const anomalies = s.anomalies.map((x) => ({ ...x, review_status: liveStatus[x.id] ?? x.review_status }));
  const pct = Math.min(100, Math.max(0, (plan.expected_savings / (s.savings_goal || 1)) * 100));
  return (
    <>
      <h2>Home Dashboard</h2>
      <DailyExpenses ov={ov} act={act} busy={busy} />
      <p className="muted">Latest month: <b>{s.current_month}</b> · Forecast month: <b>{fc.target_month}</b> · Decision: <Badge status={a.status} /></p>
      <div className="grid">
        <Card label="Monthly income" value={inr(s.income)} sub={s.income_source} />
        <Card label={`Expenses (${s.current_month})`} value={inr(s.current_spending)} />
        <Card label="Remaining balance" value={inr(s.income - s.current_spending)} />
        <Card label="Predicted next month" value={inr(fc.predicted)} sub={fc.method === 'ml_ridge' ? 'ML (Ridge)' : 'rule-based fallback'} />
        <Card label="Savings goal" value={inr(s.savings_goal)} sub={`limit ${inr(a.spending_limit)}`} />
        <Card label="Unusual transactions" value={ov.run.n_flagged} sub={`${s.anomalies.length} unusually high`} />
      </div>
      <h3>Income and expenses by month</h3>
      <Chart data={ov.monthly}>
        <XAxis dataKey="month" {...axis} /><YAxis {...axis} />
        <Bar dataKey="income" name="Income" fill="var(--brand)" /><Bar dataKey="expense" name="Expenses" fill="var(--accent)" />
      </Chart>
      <div className="grid2">
        <div><h3>Spending by category ({s.current_month})</h3>
          <Chart data={ov.categories}><XAxis dataKey="category" {...axis} /><YAxis {...axis} /><Bar dataKey="amount" name="Spent" fill="var(--brand)" /></Chart></div>
        <div><h3>Forecast</h3><ForecastChart fc={fc} /></div>
      </div>
      <div className="grid2">
        <div><h3>Savings with the recommended budget</h3>
          <div className="panel"><div className="bar"><i style={{ width: pct + '%' }} /></div>
            <p>Expected savings <b>{inr(plan.expected_savings)}</b> against a goal of <b>{inr(s.savings_goal)}</b>.</p></div></div>
        <div><h3>Active alerts ({alerts?.length ?? 0})</h3>
          <div className="panel">{alerts?.length ? alerts.slice(0, 4).map((x) => <div key={x.id}><span className={'tag ' + x.severity}>{x.severity}</span>{x.title}</div>) : <span className="muted">No active alerts.</span>}</div></div>
      </div>
      {anomalies.length > 0 && <><h3>Unusually high transactions</h3>
        <Table rows={anomalies} cols={['date', 'description', 'category', 'amount', 'typical_amount', 'anomaly_score', 'review_status'].map((key) => ({ key }))} /></>}
    </>
  );
}

// ------------------------------------------------------------------ 2. Upload
export function Upload({ ov, act, busy, tick }) {
  const [meta] = useGet('/meta');
  const [tx] = useGet('/transactions', tick);
  const [file, setFile] = useState(null);
  const [replace, setReplace] = useState(false);
  const [result, setResult] = useState(null);

  const doUpload = () => act(async () => {
    const fd = new FormData(); fd.append('file', file); fd.append('replace', replace);
    const out = await request('/upload', { method: 'POST', body: fd }); setResult(out);
    if (!out.ok) throw new Error(out.errors.join(' '));
  }, 'CSV imported and analysed.');

  return (
    <>
      <h2>Upload Transactions</h2>
      <DailyExpenses ov={ov} act={act} busy={busy} />
      <h3>Option A: demonstration scenarios</h3>
      <p className="muted">Each scenario resets the database, imports its CSV and runs the full pipeline. Every output is computed live.</p>
      <div className="grid2">{meta && Object.entries(meta.scenarios).map(([key, sc]) => (
        <div className="panel" key={key}><b>{sc.title}</b><p className="muted">{sc.story}</p>
          <p className="muted">File: data/{sc.file} · goal {inr(sc.savings_goal)}</p>
          <button className="btn" disabled={busy} onClick={() => act(async () => { const o = await post('/scenarios/' + key); setResult(null); },
            `Loaded “${sc.title}”.`)}>Load scenario</button></div>))}</div>
      <h3>Option B: upload your own CSV</h3>
      <p className="muted">Required columns: Date, Description, Amount, Type (Income/Expense). Optional: Transaction ID, Category.</p>
      <div className="row">
        <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
        <label className="row"><input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} /> Clear existing data first</label>
        <button className="btn" disabled={!file || busy} onClick={doUpload}>Validate and import</button>
        <a href="/api/transactions.csv">Download current data</a>
      </div>
      {result?.warnings?.map((w) => <div className="flash" style={{ background: 'var(--warnbg)', color: 'var(--warn)', marginTop: 10 }} key={w}>{w}</div>)}
      <h3>Current data ({ov.n_transactions})</h3>
      <Table rows={tx} cols={['txn_id', 'date', 'description', 'amount', 'type', 'category', 'category_source'].map((key) => ({ key }))} />
      {ov.n_transactions > 0 && <p><button className="btn ghost" disabled={busy} onClick={() => act(() => send('DELETE', '/data'), 'All data cleared.')}>🗑️ Clear all data</button></p>}
    </>
  );
}

// ------------------------------------------------------------------ 3. Classification
export function Classification({ ov, act, busy, tick }) {
  const [rows] = useGet('/transactions?type=Expense', tick);
  const [meta] = useGet('/meta');
  const [onlyReview, setOnlyReview] = useState(false);
  if (!ov.run) return <><h2>Expense Classification</h2><NeedData /></>;
  const exp = rows || [];
  const view = onlyReview ? exp.filter((r) => r.needs_review === 1) : exp;
  const n = (f) => exp.filter(f).length;
  return (
    <>
      <h2>Expense Classification</h2>
      <p className="step">TF-IDF → Logistic Regression. Confidence below 0.60 marks a transaction for review. Categories you correct are never overwritten.</p>
      <div className="grid">
        <Card label="Expenses" value={exp.length} /><Card label="Predicted by ML" value={n((r) => r.category_source === 'model')} />
        <Card label="Low confidence" value={n((r) => r.needs_review === 1)} /><Card label="Corrected by you" value={n((r) => r.category_source === 'corrected')} />
      </div>
      <p><label className="row"><input type="checkbox" checked={onlyReview} onChange={(e) => setOnlyReview(e.target.checked)} /> Show only low-confidence predictions</label></p>
      <Table rows={view} cols={[
        { key: 'id' }, { key: 'date' }, { key: 'description' }, { key: 'amount' },
        { key: 'category', label: 'category (change to correct)', render: (r) => meta && (
          <select value={r.category || ''} disabled={busy} aria-label={'Category for ' + r.description}
                  onChange={(e) => act(() => post(`/transactions/${r.id}/category`, { category: e.target.value }), `Transaction #${r.id} set to ${e.target.value}; analysis updated.`)}>
            {meta.categories.map((c) => <option key={c}>{c}</option>)}</select>) },
        { key: 'category_source', label: 'source' }, { key: 'predicted_category', label: 'predicted' },
        { key: 'category_confidence', label: 'confidence' }, { key: 'needs_review', label: 'review?', render: (r) => (r.needs_review ? 'yes' : '') },
      ]} />
    </>
  );
}

// ------------------------------------------------------------------ 4. Anomalies
export function Anomalies({ ov, act, busy, tick }) {
  const [rows] = useGet('/transactions?flagged=true', tick);
  if (!ov.run) return <><h2>Anomaly Detection</h2><NeedData /></>;
  return (
    <>
      <h2>Anomaly Detection</h2>
      <p className="step">Isolation Forest on expense transactions: log amount, excess over the typical amount for the category, day of week. Higher score = more unusual.</p>
      <p className="muted">{ov.run.anomaly_message}</p>
      <h3>Flagged transactions ({rows?.length ?? 0})</h3>
      {rows?.length === 0 && <div className="empty">No unusual transactions found.</div>}
      {rows?.map((r) => (
        <div className="panel" key={r.id} style={{ marginBottom: 10 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <div><b>{r.description}</b> · {r.date} · {r.category}<br /><i className="muted">{r.anomaly_explanation}</i></div>
            <div><b>{inr(r.amount)}</b><div className="muted">typical {inr(r.typical_amount)} · score {r.anomaly_score?.toFixed(3)}</div></div>
            <div>Status: <b>{r.review_status}</b><br />
              {r.review_status !== 'REVIEWED'
                ? <button className="btn" disabled={busy} onClick={() => act(() => post(`/transactions/${r.id}/review`, { reviewed: true }), 'Marked as reviewed; related alerts resolved automatically.')}>✔ Confirm as legitimate</button>
                : <button className="btn ghost" disabled={busy} onClick={() => act(() => post(`/transactions/${r.id}/review`, { reviewed: false }), 'Review re-opened.')}>↺ Re-open review</button>}
            </div>
          </div>
        </div>))}
    </>
  );
}

// ------------------------------------------------------------------ 5. Prediction
export function Prediction({ ov }) {
  if (!ov.run) return <><h2>Spending Prediction</h2><NeedData /></>;
  const fc = ov.run.forecast;
  const method = { ml_ridge: 'Machine learning (Ridge Regression)', rule_average: 'Rule-based fallback (not ML)', none: 'No forecast' }[fc.method];
  const bt = (fc.history || []).filter((h) => h.predicted != null);
  const mae = bt.length ? bt.reduce((t, h) => t + Math.abs(h.predicted - h.regular), 0) / bt.length : null;
  return (
    <>
      <h2>Spending Prediction</h2>
      <div className="grid">
        <Card label={`Predicted spending for ${fc.target_month}`} value={inr(fc.predicted)} /><Card label="Method" value={method} />
        <Card label="Backtest MAE on your data" value={mae == null ? 'n/a' : inr(mae)} sub={`${bt.length} month(s) with 3 months of history`} />
      </div>
      <p className="step">{fc.note}</p>
      <ForecastChart fc={fc} />
      <p className="muted">Regular spending = all expenses minus transactions flagged as unusually high. Backtest points use only the three months before each month.</p>
      <Table rows={fc.history} />
    </>
  );
}

// ------------------------------------------------------------------ 6. Advisor
function SettingsForm({ initial, categories, act, busy }) {
  const [f, setF] = useState(initial);
  const set = (k) => (e) => setF({ ...f, [k]: +e.target.value });
  const toggle = (c) => setF({ ...f, essential_categories: f.essential_categories.includes(c) ? f.essential_categories.filter((x) => x !== c) : [...f.essential_categories, c] });
  return (
    <div className="panel">
      <b>Your goals and constraints</b>
      <div className="row" style={{ margin: '10px 0' }}>
        <label className="f">Monthly income (₹, 0 = detect)<input type="number" min="0" step="1000" value={f.monthly_income} onChange={set('monthly_income')} /></label>
        <label className="f">Savings goal per month (₹)<input type="number" min="0" step="500" value={f.savings_goal} onChange={set('savings_goal')} /></label>
        <label className="f">Spending cap (₹, 0 = none)<input type="number" min="0" step="1000" value={f.monthly_budget} onChange={set('monthly_budget')} /></label>
      </div>
      <div className="muted">Essential categories (the agent never cuts these)</div>
      <div className="row" style={{ margin: '6px 0 12px' }}>{categories.map((c) => (
        <label key={c} className="row"><input type="checkbox" checked={f.essential_categories.includes(c)} onChange={() => toggle(c)} />{c}</label>))}</div>
      <button className="btn" disabled={busy} onClick={() => act(() => send('PUT', '/settings', f), 'Settings saved; the agent re-planned and alerts were updated.')}>Save and run the agent</button>
    </div>
  );
}

export function Advisor({ ov, act, busy, tick }) {
  const [meta] = useGet('/meta', tick);
  const run = ov.run;
  return (
    <>
      <h2>AI Financial Advisor</h2>
      <p className="muted">Goal-based agent: perceive → evaluate goals → generate plan → execute actions. All plans are simulated; no money is moved.</p>
      {meta && <SettingsForm key={JSON.stringify(meta.settings)} initial={meta.settings} categories={meta.categories} act={act} busy={busy} />}
      {!run ? <NeedData /> : (() => {
        const { state: s, assessment: a, plan } = run;
        return (<>
          <h3>1 · Perceive: current financial state</h3>
          <div className="grid">
            <Card label="Income" value={inr(s.income)} sub={s.income_source} /><Card label={`Spent in ${s.current_month}`} value={inr(s.current_spending)} />
            <Card label={`Predicted ${s.forecast_month}`} value={inr(s.predicted_spending)} sub={{ ml_ridge: 'ML (Ridge)', rule_average: 'rule-based fallback' }[s.forecast_method] ?? s.forecast_method} />
            <Card label="Savings goal" value={inr(s.savings_goal)} /><Card label="Unusually high txns" value={s.anomalies.length} />
          </div>
          <h3>2 · Evaluate goals</h3>
          <p><Badge status={a.status} /> {a.reason}</p>
          <div className="grid">
            <Card label="Spending limit" value={inr(a.spending_limit)} sub={a.limit_reason} /><Card label="Required reduction" value={inr(Math.max(0, a.gap))} />
            <Card label="Max possible reduction" value={inr(a.max_possible_reduction)} sub="flexible categories only" /><Card label="Max achievable savings" value={inr(a.max_achievable_savings)} />
          </div>
          <h3>Options evaluated</h3>
          <Table rows={a.options.map((o) => ({ ...o, max_cut_fraction: Math.round(o.max_cut_fraction * 100) + '%', essential: o.essential ? 'yes' : '' }))} />
          <h3>3 · Generate plan</h3>
          {plan.reductions.length ? <ul>{plan.reductions.map((r) => <li key={r.category}><b>{r.category}</b>: {inr(r.planned)} → <b>{inr(r.revised)}</b> (−{inr(r.reduction)}). <i className="muted">{r.reason}</i></li>)}</ul> : <p>No reductions needed.</p>}
          <Chart data={plan.revised_budget}><XAxis dataKey="category" {...axis} /><YAxis {...axis} /><Bar dataKey="planned" name="Planned" fill="var(--mute)" /><Bar dataKey="revised" name="Revised" fill="var(--brand)" /></Chart>
          <p className="muted">Revised total {inr(plan.revised_total)}; expected savings {inr(plan.expected_savings)} vs goal {inr(s.savings_goal)}.</p>
          <h3>4 · Execute actions (this run)</h3>
          <Table rows={run.action_results} />
        </>);
      })()}
    </>
  );
}

// ------------------------------------------------------------------ 7. Alerts and reports
export function Alerts({ ov, act, busy, tick }) {
  const [tab, setTab] = useState('Alerts');
  const [alerts] = useGet('/alerts', tick);
  const [props] = useGet('/proposals', tick);
  const [recs] = useGet('/recommendations', tick);
  const [log] = useGet('/actions', tick);
  const [report] = useGet('/report', tick);
  if (!ov.run) return <><h2>Alerts and Reports</h2><NeedData /></>;
  const open = (alerts || []).filter((a) => a.status !== 'RESOLVED');
  const done = (alerts || []).filter((a) => a.status === 'RESOLVED');
  const latest = props?.latest;
  return (
    <>
      <h2>Alerts and Reports</h2>
      <div className="tabs" role="tablist">{['Alerts', 'Budget proposals', 'Recommendations', 'Action log', 'Export'].map((t) => (
        <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>{t}</button>))}</div>
      {tab === 'Alerts' && <>
        {!open.length && <div className="empty">No open alerts.</div>}
        {open.map((a) => (<div className="panel" key={a.id} style={{ marginBottom: 10 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <div><span className={'tag ' + a.severity}>{a.severity}</span><b>{a.title}</b> <code>{a.status}</code><p>{a.message}</p>
              <span className="muted">{a.alert_type} · updated {a.updated_at}</span></div>
            {a.status === 'ACTIVE' && <button className="btn ghost" disabled={busy} onClick={() => act(() => post(`/alerts/${encodeURIComponent(a.alert_key)}/acknowledge`), 'Alert acknowledged.')}>Acknowledge</button>}
          </div></div>))}
        {done.length > 0 && <><h3>Resolved ({done.length})</h3><Table rows={done} cols={['severity', 'title', 'resolved_reason', 'updated_at'].map((key) => ({ key }))} /></>}
      </>}
      {tab === 'Budget proposals' && <>
        {latest && <div className="panel"><b>Latest proposal #{latest.id}</b> for {latest.target_month} · {latest.feasibility} · <code>{latest.status}</code>
          <p>Limit {inr(latest.spending_limit)}, predicted {inr(latest.predicted_spending)}, revised total {inr(latest.revised_total)}.</p>
          {latest.status === 'PROPOSED' && <button className="btn" disabled={busy} onClick={() => act(() => post(`/proposals/${latest.id}/accept`), 'Budget proposal accepted.')}>Accept this budget proposal</button>}</div>}
        <h3>All proposals</h3><Table rows={props?.all} /></>}
      {tab === 'Recommendations' && (recs?.length ? <ul>{recs.map((r) => <li key={r.id}>{r.message}</li>)}</ul> : <p className="muted">No active recommendations.</p>)}
      {tab === 'Action log' && <Table rows={log} />}
      {tab === 'Export' && <>
        <div className="row"><a className="btn" href="/api/report.md">⬇️ Download report (Markdown)</a><a className="btn ghost" href="/api/transactions.csv">⬇️ Download transactions (CSV)</a></div>
        <h3>Report preview</h3><pre className="md">{report?.markdown}</pre></>}
    </>
  );
}
