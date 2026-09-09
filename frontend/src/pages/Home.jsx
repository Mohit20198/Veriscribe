import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';

function StatTile({ value, label }) {
  return (
    <div className="stat-tile">
      <div className="stat-tile-value">{value ?? '—'}</div>
      <div className="stat-tile-label">{label}</div>
    </div>
  );
}

export default function Home() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        // Fetch raw response to get relationships_by_type at top level
        const raw = await fetch(`${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/documents`);
        const res = await raw.json();
        const docs = res.documents ?? [];
        const totalFacts = res.total_facts ?? docs.reduce((s, d) => s + (d.fact_count ?? 0), 0);
        const rel = res.relationships_by_type ?? {};
        setStats({
          totalDocs: docs.length,
          totalFacts,
          corroborates: rel.corroborates ?? 0,
          contradicts:  rel.contradicts  ?? 0,
          reconciled:   rel.reconciled   ?? 0,
          ambiguous:    rel.ambiguous    ?? 0,
        });
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <div className="page">
      {/* Hero */}
      <div style={{ maxWidth: 680, marginBottom: 'var(--sp-2xl)' }}>
        <h1 style={{ marginBottom: 'var(--sp-base)' }}>
          Every fact. Every source.<br />Every relationship — verified.
        </h1>
        <p style={{ fontSize: 17, color: 'var(--c-text-muted)', marginBottom: 'var(--sp-xl)', lineHeight: 1.7 }}>
          Veriscribe extracts structured facts from financial PDFs, grounds each one
          in its source evidence, and automatically detects corroborations, contradictions,
          and reconciliations across documents.
        </p>
        <Link to="/explore" className="btn btn-primary" style={{ fontSize: 15, padding: '10px 24px' }}>
          Explore Facts
        </Link>
        &nbsp;&nbsp;
        <Link to="/documents" className="btn btn-outline" style={{ fontSize: 15, padding: '10px 24px' }}>
          View Documents
        </Link>
      </div>

      <hr className="divider" />

      {/* Stats */}
      <h3 style={{ marginBottom: 'var(--sp-sm)' }}>Knowledge Base</h3>
      {loading && <div className="loading">Loading stats…</div>}
      {error   && <div className="error-msg">Could not load stats — is the API running? ({error})</div>}
      {stats && (
        <div className="stat-grid">
          <StatTile value={stats.totalDocs}    label="Documents" />
          <StatTile value={stats.totalFacts}   label="Facts extracted" />
          <StatTile value={stats.corroborates} label="Corroborations" />
          <StatTile value={stats.contradicts}  label="Contradictions" />
          <StatTile value={stats.reconciled}   label="Reconciled" />
        </div>
      )}

      <hr className="divider" style={{ marginTop: 'var(--sp-2xl)' }} />

      {/* How it works */}
      <h3 style={{ marginBottom: 'var(--sp-lg)' }}>How it works</h3>
      <div className="how-strip">
        {[
          { n: '1', title: 'Upload', body: 'Drop any financial PDF — annual report, earnings deck, prospectus — and the system parses every page.' },
          { n: '2', title: 'Extract & Ground', body: 'An LLM extracts structured facts from text, tables, and charts. Each fact is anchored to its source evidence.' },
          { n: '3', title: 'Compare & Reason', body: 'Facts are canonicalized, embedded, and compared. The judge determines whether cross-document facts corroborate, contradict, reconcile, or are ambiguous.' },
        ].map(s => (
          <div className="how-step" key={s.n}>
            <div className="how-step-num">{s.n}</div>
            <div className="how-step-body">
              <h4>{s.title}</h4>
              <p>{s.body}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
