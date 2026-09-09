import { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { api } from '../api';
import FactCard from '../components/FactCard';

export default function DocumentDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [facts, setFacts] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setFacts(null);
    setError(null);
    api.getFacts({ document_id: id, limit: 200 })
      .then(data => setFacts(Array.isArray(data) ? data : []))
      .catch(e => setError(e.message));
  }, [id]);

  return (
    <div className="page">
      <div style={{ marginBottom: 'var(--sp-base)' }}>
        <Link to="/documents" style={{ color: 'var(--c-text-faint)', fontSize: 13 }}>← Documents</Link>
      </div>

      <div className="doc-header">
        <h2>{id}</h2>
        <div className="doc-meta">
          {facts && <span>{facts.length} facts</span>}
        </div>
      </div>

      {!facts && !error && <div className="loading">Loading facts…</div>}
      {error && <div className="error-msg">Could not load facts — {error}</div>}

      {facts && facts.length === 0 && (
        <p style={{ color: 'var(--c-text-faint)' }}>No facts found for this document.</p>
      )}

      {facts && facts.map(fact => (
        <FactCard
          key={fact.fact_id}
          fact={fact}
          onClick={() => navigate(`/explore/${fact.fact_id}`)}
        />
      ))}
    </div>
  );
}
