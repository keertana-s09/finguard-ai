import { useEffect, useState } from 'react';

export const inr = (n) => '₹' + Math.round(Number(n) || 0).toLocaleString('en-IN');

const parse = async (r) => {
  if (!r.ok) {
    const b = await r.json().catch(() => ({}));
    throw new Error(typeof b.detail === 'string' ? b.detail : r.statusText);
  }
  return r.json();
};
export const get = (p) => fetch('/api' + p).then(parse);
export const send = (method, p, body) =>
  fetch('/api' + p, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined }).then(parse);

/** GET a path; reloads when `dep` changes. Returns [data, error]. */
export function useGet(path, dep) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    let live = true;
    setErr(null);
    get(path).then((d) => live && setData(d)).catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, [path, dep]);
  return [data, err];
}

export const Card = ({ label, value, sub }) => (
  <div className="card"><div className="l">{label}</div><div className="v">{value}</div>{sub && <div className="s">{sub}</div>}</div>
);

const STATUS = { ON_TRACK: ['On track', 'ok'], ACHIEVABLE_WITH_CUTS: ['Achievable with cuts', 'warn'], INFEASIBLE: ['Goal infeasible', 'bad'] };
export const Badge = ({ status }) => {
  const [text, cls] = STATUS[status] || [status, 'warn'];
  return <span className={'badge ' + cls}>{text}</span>;
};

const fmt = (v) => (v == null ? '' : typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : String(v));

/** Generic table. `cols` is optional: without it, columns come from the first row's keys. */
export function Table({ rows, cols }) {
  if (!rows || !rows.length) return <p className="muted">Nothing to show.</p>;
  const columns = cols || Object.keys(rows[0]).filter((k) => typeof rows[0][k] !== 'object').map((key) => ({ key }));
  return (
    <div className="tbl"><table>
      <thead><tr>{columns.map((c) => <th key={c.key}>{c.label || c.key}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => (
        <tr key={r.id ?? i}>{columns.map((c) => <td key={c.key}>{c.render ? c.render(r) : fmt(r[c.key])}</td>)}</tr>
      ))}</tbody>
    </table></div>
  );
}

export const NeedData = () => <div className="empty">No analysed data yet. Open <b>Upload Transactions</b> and load a demo scenario or a CSV.</div>;
