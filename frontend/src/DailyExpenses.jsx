import { useState } from 'react';
import { Card, send, useGet } from './ui.jsx';

const money = (value) => '₹' + Number(value).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const localDate = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};
const emptyEntry = () => ({ date: localDate(), description: '', amount: '', category: '', entry_id: crypto.randomUUID() });

export default function DailyExpenses({ ov, act, busy }) {
  const [meta] = useGet('/meta');
  const [entry, setEntry] = useState(emptyEntry);
  const set = (key) => (e) => setEntry((old) => ({ ...old, [key]: e.target.value }));
  const budget = ov.monthly_budget;
  const save = (e) => {
    e.preventDefault();
    act(async () => {
      const result = await send('POST', '/expenses', { ...entry, amount: Number(entry.amount), category: entry.category || null });
      setEntry(emptyEntry());
      if (!result.analysis_ok) throw new Error('Expense saved, but analysis could not refresh. Re-run analysis; do not enter it again.');
    }, 'Expense saved. Your remaining monthly budget is updated.');
  };
  return <section>
    <h3>Add a daily expense</h3>
    <form className="panel" onSubmit={save}>
      <fieldset disabled={busy} style={{ border: 0, padding: 0, margin: 0 }}>
        <div className="row">
          <label className="f">Expense date<input type="date" required max={localDate()} value={entry.date} onChange={set('date')} /></label>
          <label className="f">Expense amount (₹)<input type="number" required min="0.01" step="0.01" value={entry.amount} onChange={set('amount')} /></label>
          <label className="f">Expense description<input required maxLength={500} value={entry.description} onChange={set('description')} placeholder="e.g. Lunch at college" /></label>
          <label className="f">Expense category<select value={entry.category} onChange={set('category')}>
            <option value="">Predict automatically</option>
            {meta?.categories.map((category) => <option key={category}>{category}</option>)}
          </select></label>
          <button className="btn" type="submit">Save expense</button>
        </div>
      </fieldset>
    </form>
    {budget && <>
      <h3>Remaining monthly budget ({budget.month})</h3>
      <div className="grid">
        <Card label="Monthly spending limit" value={budget.spending_limit == null ? 'Not set' : money(budget.spending_limit)} />
        <Card label="Recorded expenses this month" value={money(budget.spent)} />
        <Card label="Remaining budget" value={budget.remaining == null ? 'Not set' : money(budget.remaining)} />
      </div>
      <p role="status">{budget.recommendation}</p>
      {budget.suggestions.map((suggestion) => <p key={suggestion}>{suggestion}</p>)}
    </>}
  </section>;
}
