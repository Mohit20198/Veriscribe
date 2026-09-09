import { useEffect, useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';

function StatusBadge({ status }) {
  const cls = status === 'done' ? 'badge-done' : status === 'error' ? 'badge-error' : 'badge-processing';
  const icon = status === 'done' ? '✓' : status === 'error' ? '✕' : '…';
  return <span className={`badge ${cls}`}>{icon} {status}</span>;
}

export default function Documents() {
  const [docs, setDocs] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  async function fetchDocs() {
    try {
      const d = await api.getDocuments();
      setDocs(d);
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => { fetchDocs(); }, []);

  async function handleFiles(files) {
    const file = files[0];
    if (!file?.name?.endsWith('.pdf')) { setError('Please select a PDF file.'); return; }
    setError(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await api.uploadDocument(fd);
      // Optimistically add to list as processing
      setDocs(d => [{ document_id: res.document_id, filename: file.name, status: 'processing', fact_count: 0 }, ...(d ?? [])]);
      // Poll status for this doc
      let attempts = 0;
      const docId = res.document_id;
      const timer = setInterval(async () => {
        attempts++;
        if (attempts > 60) { clearInterval(timer); return; }
        try {
          const s = await api.getDocumentStatus(docId);
          setDocs(prev => prev.map(d => d.document_id === docId ? { ...d, status: s.status, fact_count: s.fact_count ?? d.fact_count } : d));
          if (s.status === 'done' || s.status === 'error') clearInterval(timer);
        } catch {}
      }, 3000);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="page">
      <h2 style={{ marginBottom: 'var(--sp-lg)' }}>Documents</h2>

      {/* Dropzone */}
      <div
        className={`dropzone ${dragging ? 'drag-over' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
      >
        <input ref={inputRef} type="file" accept=".pdf" onChange={e => handleFiles(e.target.files)} />
        <div className="material-symbols-outlined dropzone-icon">upload_file</div>
        <div style={{ fontWeight: 600, fontSize: 15, marginTop: 8 }}>Drop a PDF to upload</div>
        <div className="dropzone-hint">or click to browse — PDF files only</div>
      </div>

      {error && <div className="error-msg">{error}</div>}

      {/* Document table */}
      {!docs && !error && <div className="loading">Loading documents…</div>}
      {docs && docs.length === 0 && (
        <p style={{ marginTop: 'var(--sp-xl)', color: 'var(--c-text-faint)' }}>No documents ingested yet. Upload a PDF above to get started.</p>
      )}
      {docs && docs.length > 0 && (
        <table className="doc-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Status</th>
              <th>Facts</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {docs.map(doc => (
              <tr key={doc.document_id}>
                <td>
                  <span className="mono" style={{ fontSize: 12 }}>{doc.document_id}</span>
                  {doc.filename && <><br /><span className="text-faint" style={{ fontSize: 11 }}>{doc.filename}</span></>}
                </td>
                <td><StatusBadge status={doc.status ?? 'done'} /></td>
                <td className="mono">{doc.fact_count ?? '—'}</td>
                <td>
                  <Link to={`/documents/${doc.document_id}`} className="btn btn-outline btn-sm">View →</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
