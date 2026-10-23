import { useState } from 'react';
import { Badge, send, useGet } from './ui.jsx';
import { Home, Upload, Classification, Anomalies, Prediction, Advisor, Alerts } from './pages.jsx';

const PAGES = [
  ['🏠 Home Dashboard', Home], ['📤 Upload Transactions', Upload], ['🏷️ Expense Classification', Classification],
  ['🚨 Anomaly Detection', Anomalies], ['📈 Spending Prediction', Prediction], ['🤖 AI Financial Advisor', Advisor],
  ['🔔 Alerts and Reports', Alerts],
];

export default function App() {
  const start = Math.min(Math.max((+new URLSearchParams(location.search).get('page') || 1) - 1, 0), PAGES.length - 1);
  const [page, setPage] = useState(start);
  const [tick, setTick] = useState(0);
  const [flash, setFlash] = useState(null);
  const [busy, setBusy] = useState(false);
  const [ov, err] = useGet('/overview', tick);

  /** Run an API call, show a message, then reload all data. */
  const act = async (fn, message) => {
    setBusy(true);
    try { await fn(); setFlash({ text: message }); } catch (e) { setFlash({ text: e.message, error: true }); }
    setBusy(false);
    setTick((t) => t + 1);
  };

  const Page = PAGES[page][1];
  const run = ov?.run;
  return (
    <div className="layout">
      <nav>
        <h1>🛡️ FinGuard AI</h1>
        <p className="sub">Personal finance and budget agent</p>
        {PAGES.map(([name], i) => (
          <button key={name} className="item" aria-current={i === page ? 'page' : undefined}
                  onClick={() => { setPage(i); setFlash(null); }}>{name}</button>
        ))}
        {ov && (
          <div className="stats">
            <div>Transactions: <b>{ov.n_transactions}</b></div>
            <div>Active alerts: <b>{ov.n_active_alerts}</b></div>
            {run && <div style={{ marginTop: 6 }}><Badge status={run.assessment.status} /></div>}
            <button className="btn ghost" style={{ marginTop: 10 }} disabled={busy || !ov.n_transactions}
                    onClick={() => act(() => send('POST', '/analysis/run'), 'Analysis updated.')}>
              🔄 Re-run analysis
            </button>
          </div>
        )}
      </nav>
      <main>
        {flash && <div className={'flash' + (flash.error ? ' err' : '')} role="status">{flash.text}</div>}
        {err && (err.unreachable
          ? <div className="flash err">Cannot reach the API. Is <code>uvicorn api:app --port 8000</code> running?</div>
          : <div className="flash err">The API returned an error ({err.status}: {err.message}). Check the uvicorn terminal for details.</div>)}
        {busy && <p className="muted">Running classifier → Isolation Forest → forecast → agent → automated response…</p>}
        {ov && <Page ov={ov} act={act} busy={busy} tick={tick} />}
      </main>
    </div>
  );
}
