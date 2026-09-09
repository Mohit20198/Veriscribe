import { useState, useRef } from 'react';
import { api } from '../api';

// Poll until status = done or error
async function pollStatus(docId, onUpdate) {
  let attempts = 0;
  while (attempts < 60) {
    await new Promise(r => setTimeout(r, 3000));
    attempts++;
    try {
      const status = await api.getDocumentStatus(docId);
      onUpdate(status);
      if (status.status === 'done' || status.status === 'error') return status;
    } catch { /* keep polling on transient error */ }
  }
  return null;
}

export default function UploadModal({ onClose, onDone }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(null);   // { status, message }
  const [error, setError] = useState(null);
  const inputRef = useRef();

  function handleFile(f) {
    if (!f?.name?.endsWith('.pdf')) { setError('Please select a PDF file.'); return; }
    setFile(f);
    setError(null);
  }

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setError(null);
    setProgress({ status: 'uploading', message: 'Uploading…' });

    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await api.uploadDocument(fd);
      const docId = res.document_id;

      setProgress({ status: 'processing', message: 'Processing — extracting facts…' });
      const final = await pollStatus(docId, s => {
        setProgress({ status: s.status, message: `Processing (${s.fact_count ?? 0} facts found so far)…` });
      });

      if (final?.status === 'done') {
        setProgress({ status: 'done', message: `Done — ${final.fact_count} facts extracted.` });
        setTimeout(() => onDone(docId), 800);
      } else {
        setError(final?.error ?? 'Ingestion timed out or failed.');
        setUploading(false);
      }
    } catch (e) {
      setError(e.message);
      setUploading(false);
      setProgress(null);
    }
  }

  return (
    <div style={{ position:'fixed', inset:0, background:'rgba(0,0,0,0.45)', zIndex:100, display:'flex', alignItems:'center', justifyContent:'center' }}>
      <div style={{ background:'#fff', borderRadius:8, padding:32, width:480, position:'relative' }}>
        <button onClick={onClose} style={{ position:'absolute', top:16, right:16, background:'none', border:'none', fontSize:20, cursor:'pointer', color:'var(--c-text-faint)' }}>✕</button>
        <h3 style={{ marginBottom:16 }}>Upload PDF</h3>

        <div
          className={`dropzone ${dragging ? 'drag-over' : ''}`}
          onClick={() => !uploading && inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        >
          <input ref={inputRef} type="file" accept=".pdf" onChange={e => handleFile(e.target.files[0])} />
          <div className="material-symbols-outlined dropzone-icon">upload_file</div>
          <div style={{ marginTop:8, fontWeight:600, fontSize:14 }}>
            {file ? file.name : 'Drop PDF here or click to browse'}
          </div>
          <div className="dropzone-hint">Only PDF files are accepted</div>
        </div>

        {error && <div className="error-msg">{error}</div>}

        {progress && (
          <div className="upload-progress">
            <span className={`badge badge-${progress.status === 'done' ? 'done' : progress.status === 'error' ? 'error' : 'processing'}`}>
              {progress.status}
            </span>
            &nbsp;{progress.message}
          </div>
        )}

        <div style={{ display:'flex', justifyContent:'flex-end', gap:8, marginTop:16 }}>
          <button className="btn btn-outline" onClick={onClose} disabled={uploading}>Cancel</button>
          <button className="btn btn-primary" onClick={handleUpload} disabled={!file || uploading}>
            {uploading ? 'Processing…' : 'Upload & Ingest'}
          </button>
        </div>
      </div>
    </div>
  );
}
