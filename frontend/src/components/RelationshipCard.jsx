import { api } from '../api';

const REL_META = {
  corroborates: { icon: 'check_circle', label: 'Corroborates' },
  contradicts:  { icon: 'cancel',       label: 'Contradicts'  },
  reconciled:   { icon: 'balance',      label: 'Reconciled'   },
  ambiguous:    { icon: 'help',         label: 'Ambiguous'    },
  unrelated:    { icon: 'remove_circle',label: 'Unrelated'    },
};

export default function RelationshipCard({ rel }) {
  const meta = REL_META[rel.relationship_type] ?? { icon: 'help', label: rel.relationship_type };
  const confidence = rel.confidence != null
    ? `${(rel.confidence * 100).toFixed(0)}%`
    : null;

  return (
    <div className="rel-card">
      <div className="rel-header">
        <span className="rel-pill">
          <span className="material-symbols-outlined" style={{ fontSize: 13 }}>{meta.icon}</span>
          {meta.label}
        </span>
        {confidence && <span className="rel-confidence">{confidence}</span>}
      </div>
      {rel.related_fact && (
        <div className="rel-statement">{rel.related_fact.statement}</div>
      )}
      {rel.justification && (
        <div className="rel-justification">{rel.justification}</div>
      )}
      {rel.related_fact && (
        <div className="rel-doc">{rel.related_fact.document_id}</div>
      )}
    </div>
  );
}
