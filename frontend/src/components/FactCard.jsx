// Shared FactCard — used in both Fact Browser (left panel) and Document Detail list.
import { API_BASE } from '../api';

const SOURCE_ICONS = {
  text: 'article',
  table: 'table_chart',
  chart: 'bar_chart',
};

// Fill between 0 and 5 pips based on confidence 0–1
function ConfidenceBar({ value }) {
  const filled = Math.round((value ?? 0) * 5);
  return (
    <div className="confidence-bar" title={`Confidence: ${((value ?? 0) * 100).toFixed(0)}%`}>
      {[0,1,2,3,4].map(i => (
        <div key={i} className={`confidence-pip ${i < filled ? 'filled' : ''}`} />
      ))}
    </div>
  );
}

export default function FactCard({ fact, active, onClick }) {
  const sourceType = fact.source_type ?? fact.chunk_type ?? 'text';
  const icon = SOURCE_ICONS[sourceType] ?? 'article';

  return (
    <div className={`fact-card ${active ? 'active' : ''}`} onClick={onClick}>
      <div className="fact-card-header">
        <span className="fact-card-doc">{fact.document_id}</span>
        <span className="material-symbols-outlined fact-card-type-icon">{icon}</span>
      </div>
      <div className="fact-card-statement">{fact.statement}</div>
      <ConfidenceBar value={fact.confidence} />
    </div>
  );
}
