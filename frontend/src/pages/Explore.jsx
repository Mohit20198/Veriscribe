import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api';
import FactCard from '../components/FactCard';
import RelationshipCard from '../components/RelationshipCard';

// ── Detail panel ────────────────────────────────────────────────────────────
function FactDetail({ factId }) {
  const [fact, setFact] = useState(null);
  const [rels, setRels] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!factId) return;
    setLoading(true);
    setFact(null);
    setRels(null);
    setError(null);

    Promise.all([api.getFact(factId), api.getRelationships(factId)])
      .then(([f, r]) => { setFact(f); setRels(r); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [factId]);

  if (!factId) return <div className="detail-empty">← Select a fact to inspect</div>;
  if (loading) return <div className="loading">Loading fact detail…</div>;
  if (error)   return <div className="detail-panel"><div className="error-msg">Could not load fact — {error}</div></div>;

  const isChart = fact.chunk_type === 'chart' || fact.source_type === 'chart';
  const imageUrl = api.getEvidenceImageUrl(factId);

  return (
    <div className="detail-panel">
      <div className="tag">{fact.document_id}</div>
      <h3 style={{ marginTop: 'var(--sp-sm)', marginBottom: 'var(--sp-base)' }}>{fact.statement}</h3>

      {/* Temporal scope + attribute */}
      {(fact.raw_attribute || fact.temporal_scope) && (
        <div style={{ display: 'flex', gap: 'var(--sp-sm)', flexWrap: 'wrap', marginBottom: 'var(--sp-base)' }}>
          {fact.raw_attribute && <span className="tag">{fact.raw_attribute}</span>}
          {fact.temporal_scope && Object.keys(fact.temporal_scope).length > 0 && (
            <span className="tag">{JSON.stringify(fact.temporal_scope)}</span>
          )}
          {fact.confidence != null && (
            <span className="tag mono">{(fact.confidence * 100).toFixed(0)}% conf</span>
          )}
        </div>
      )}

      {/* Evidence */}
      <h4 style={{ marginBottom: 'var(--sp-xs)' }}>Evidence</h4>
      {fact.evidence_quote && (
        <blockquote className="evidence-block">
          "{fact.evidence_quote}"
          <div className="evidence-meta">
            {fact.source_type && `Source type: ${fact.source_type}`}
          </div>
        </blockquote>
      )}
      {fact.evidence_context_window && (
        <details style={{ marginTop: 'var(--sp-sm)' }}>
          <summary style={{ fontSize: 12, color: 'var(--c-text-faint)', cursor: 'pointer' }}>Context window</summary>
          <pre style={{ fontSize: 11, color: 'var(--c-text-muted)', whiteSpace: 'pre-wrap', marginTop: 'var(--sp-xs)', lineHeight: 1.5, background: 'var(--c-surface-low)', padding: 'var(--sp-sm)', borderRadius: 'var(--r-sm)' }}>
            {fact.evidence_context_window}
          </pre>
        </details>
      )}
      {isChart && (
        <img
          src={imageUrl}
          alt="Chart evidence"
          className="evidence-image"
          onError={e => { e.target.style.display = 'none'; }}
        />
      )}

      {/* Relationships */}
      <div className="relationships-section">
        <h4 style={{ marginBottom: 'var(--sp-sm)' }}>
          Relationships {rels && <span className="text-faint" style={{ fontWeight: 400 }}>({rels.length})</span>}
        </h4>
        {!rels && <div className="loading">Loading…</div>}
        {rels && rels.length === 0 && (
          <p style={{ color: 'var(--c-text-faint)', fontSize: 13 }}>No relationships found for this fact.</p>
        )}
        {rels && rels.map((rel, i) => <RelationshipCard key={i} rel={rel} />)}
      </div>
    </div>
  );
}

// ── Left panel — fact list ────────────────────────────────────────────────
export default function Explore() {
  const { factId } = useParams();
  const navigate = useNavigate();

  const [docs, setDocs] = useState([]);
  const [facts, setFacts] = useState(null);
  const [factsError, setFactsError] = useState(null);
  const [search, setSearch] = useState('');
  const [docFilter, setDocFilter] = useState('');
  const [selectedId, setSelectedId] = useState(factId ?? null);
  const debounceRef = useRef(null);

  // Sync URL param → selected
  useEffect(() => {
    if (factId) setSelectedId(factId);
  }, [factId]);

  // Fetch document list for filter dropdown
  useEffect(() => {
    api.getDocuments().then(setDocs).catch(() => {});
  }, []);

  const fetchFacts = useCallback(async (searchVal, docVal) => {
    setFactsError(null);
    try {
      const data = await api.getFacts({ search: searchVal, document_id: docVal, limit: 100 });
      setFacts(Array.isArray(data) ? data : data.facts ?? []);
    } catch (e) {
      setFactsError(e.message);
    }
  }, []);

  // Initial load
  useEffect(() => { fetchFacts('', ''); }, [fetchFacts]);

  // Debounced search
  function handleSearch(val) {
    setSearch(val);
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchFacts(val, docFilter), 350);
  }

  function handleDocFilter(val) {
    setDocFilter(val);
    fetchFacts(search, val);
  }

  function selectFact(id) {
    setSelectedId(id);
    navigate(`/explore/${id}`, { replace: true });
  }

  return (
    <div className="explore-layout">
      {/* LEFT */}
      <div className="explore-left">
        <div className="search-bar">
          <input
            className="search-input"
            placeholder="Search facts…"
            value={search}
            onChange={e => handleSearch(e.target.value)}
          />
          <select
            className="filter-select"
            value={docFilter}
            onChange={e => handleDocFilter(e.target.value)}
          >
            <option value="">All documents</option>
            {docs.map(d => (
              <option key={d.document_id} value={d.document_id}>{d.document_id}</option>
            ))}
          </select>
        </div>

        <div className="fact-list">
          {!facts && !factsError && <div className="loading">Loading facts…</div>}
          {factsError && <div className="error-msg" style={{ margin: 'var(--sp-base)' }}>Could not load facts — is the API running? ({factsError})</div>}
          {facts && facts.length === 0 && (
            <div style={{ padding: 'var(--sp-xl)', color: 'var(--c-text-faint)', fontSize: 13 }}>No facts match.</div>
          )}
          {facts && facts.map(f => (
            <FactCard
              key={f.fact_id}
              fact={f}
              active={f.fact_id === selectedId}
              onClick={() => selectFact(f.fact_id)}
            />
          ))}
        </div>
      </div>

      {/* RIGHT */}
      <div className="explore-right">
        <FactDetail factId={selectedId} />
      </div>
    </div>
  );
}
