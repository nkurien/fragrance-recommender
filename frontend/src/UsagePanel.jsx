import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'usageAdminKey';
const REFRESH_MS = 30000;

const readKey = () => {
  try {
    return localStorage.getItem(STORAGE_KEY) || '';
  } catch {
    return '';
  }
};

const writeKey = (value) => {
  try {
    if (value) localStorage.setItem(STORAGE_KEY, value);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Private mode / blocked storage: the key just won't persist
  }
};

const shortModel = (model) => model.split('/').pop().replace('gpt-oss-', '');
const compact = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : `${n}`);
const level = (pct) => (pct >= 90 ? 'danger' : pct >= 70 ? 'warn' : 'ok');

// Requests: Groq's own headers give the authoritative limit/remaining per model.
// Report whichever model is closest to its cap.
function requestsBar(models) {
  const known = models.filter((m) => m.limit_requests && m.remaining_requests != null);
  if (!known.length) return null;
  const worst = known
    .map((m) => ({ m, pct: ((m.limit_requests - m.remaining_requests) / m.limit_requests) * 100 }))
    .sort((a, b) => b.pct - a.pct)[0];
  const used = worst.m.limit_requests - worst.m.remaining_requests;
  return {
    pct: worst.pct,
    detail: `${used.toLocaleString()} / ${worst.m.limit_requests.toLocaleString()} · ${shortModel(worst.m.model)}`,
  };
}

// Tokens: Groq doesn't report a daily token limit, so compare our own 24h tally to the configured one.
function tokensBar(models, limit) {
  if (!models.length || !limit) return null;
  const worst = [...models].sort((a, b) => b.tokens_24h - a.tokens_24h)[0];
  return {
    pct: (worst.tokens_24h / limit) * 100,
    detail: `${compact(worst.tokens_24h)} / ${compact(limit)} · ${shortModel(worst.model)}`,
  };
}

function Bar({ label, bar }) {
  const pct = bar ? Math.min(bar.pct, 100) : 0;
  return (
    <div className="usage-bar">
      <div className="usage-bar-head">
        <span>{label}</span>
        <span className="usage-bar-detail">{bar ? bar.detail : 'No data yet'}</span>
      </div>
      <div className="usage-bar-track" role="progressbar" aria-label={label} aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
        <div className={`usage-bar-fill usage-bar-fill--${level(pct)}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

/**
 * Hidden admin panel, bottom-right. Toggle with Ctrl+Shift+U, or load the page with ?usage.
 * Needs the backend's ADMIN_KEY, which is asked for once and kept in localStorage.
 */
export default function UsagePanel() {
  const [open, setOpen] = useState(() => new URLSearchParams(window.location.search).has('usage'));
  const [adminKey, setAdminKey] = useState(readKey);
  const [draftKey, setDraftKey] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState(null);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'u') {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const load = useCallback(async (signal) => {
    const apiUrl = import.meta.env.VITE_API_URL || '';
    try {
      const res = await fetch(`${apiUrl}/api/usage`, { headers: { 'X-Admin-Key': adminKey }, signal });
      if (res.status === 401) {
        writeKey('');
        setAdminKey('');
        setData(null);
        setError('That key was rejected.');
        return;
      }
      if (res.status === 404) {
        setError('Usage tracking is disabled on this backend (no ADMIN_KEY set).');
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setUpdatedAt(new Date());
      setError('');
    } catch (err) {
      if (err.name !== 'AbortError') setError('Could not reach the backend.');
    }
  }, [adminKey]);

  // Fetch on open, then poll while visible
  useEffect(() => {
    if (!open || !adminKey) return undefined;
    const controller = new AbortController();
    const first = setTimeout(() => load(controller.signal), 0);
    const interval = setInterval(() => load(controller.signal), REFRESH_MS);
    return () => {
      controller.abort();
      clearTimeout(first);
      clearInterval(interval);
    };
  }, [open, adminKey, load]);

  if (!open) return null;

  const submitKey = (e) => {
    e.preventDefault();
    const trimmed = draftKey.trim();
    if (!trimmed) return;
    writeKey(trimmed);
    setAdminKey(trimmed);
    setDraftKey('');
    setError('');
  };

  const models = data?.models ?? [];

  return (
    <aside className="usage-panel" aria-label="LLM usage">
      <div className="usage-header">
        <span>LLM usage · 24h</span>
        <div className="usage-actions">
          {adminKey && (
            <button type="button" onClick={() => load()} aria-label="Refresh usage" title="Refresh">↻</button>
          )}
          <button type="button" onClick={() => setOpen(false)} aria-label="Close usage panel" title="Close (Esc)">✕</button>
        </div>
      </div>

      {!adminKey ? (
        <form className="usage-key-form" onSubmit={submitKey}>
          <input
            type="password"
            placeholder="Admin key"
            value={draftKey}
            onChange={(e) => setDraftKey(e.target.value)}
            autoComplete="off"
            autoFocus
          />
          <button type="submit" disabled={!draftKey.trim()}>Unlock</button>
        </form>
      ) : (
        <>
          <Bar label="Requests" bar={data && requestsBar(models)} />
          <Bar label="Tokens" bar={data && tokensBar(models, data.daily_token_limit)} />

          {models.length > 0 && (
            <ul className="usage-models">
              {models.map((m) => (
                <li key={m.model}>
                  <span>{shortModel(m.model)}</span>
                  <span>{m.requests_24h} req · {compact(m.tokens_24h)} tok · {m.avg_latency_ms}ms</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}

      {error && <p className="usage-error">{error}</p>}
      {updatedAt && !error && <p className="usage-footnote">Updated {updatedAt.toLocaleTimeString()}</p>}
    </aside>
  );
}
