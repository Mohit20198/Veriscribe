import { useEffect, useState, useCallback } from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { api } from '../api';
import UploadModal from './UploadModal';

export default function Layout() {
  const [counter, setCounter] = useState({ docs: 0, facts: 0 });
  const [showUpload, setShowUpload] = useState(false);
  const navigate = useNavigate();

  const fetchCounter = useCallback(async () => {
    try {
      const docs = await api.getDocuments();
      const totalFacts = docs.reduce((s, d) => s + (d.fact_count ?? 0), 0);
      setCounter({ docs: docs.length, facts: totalFacts });
    } catch { /* silent — counter is decorative */ }
  }, []);

  useEffect(() => {
    fetchCounter();
    const id = setInterval(fetchCounter, 30_000);
    return () => clearInterval(id);
  }, [fetchCounter]);

  function handleUploadDone(docId) {
    setShowUpload(false);
    fetchCounter();
    navigate(`/documents/${docId}`);
  }

  return (
    <div className="app-shell">
      <header className="navbar">
        <div className="navbar-left">
          <NavLink to="/" className="navbar-wordmark">Veriscribe</NavLink>
          <nav className="navbar-nav">
            <NavLink to="/" end>Home</NavLink>
            <NavLink to="/documents">Documents</NavLink>
            <NavLink to="/explore">Explore Facts</NavLink>
            <NavLink to="/about">About</NavLink>
          </nav>
        </div>
        <div className="navbar-right">
          <span className="navbar-counter">
            {counter.docs} documents · {counter.facts} facts
          </span>
          <button className="btn btn-primary" onClick={() => setShowUpload(true)}>
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>upload</span>
            Upload PDF
          </button>
        </div>
      </header>

      <main className="app-main">
        <Outlet context={{ refreshCounter: fetchCounter }} />
      </main>

      <footer className="footer">
        <span>Veriscribe — Fact Knowledge Layer &nbsp;·&nbsp; <span className="text-faint">Built for Superjoin VIT 2026</span></span>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer">
          ⌥ View source
        </a>
      </footer>

      {showUpload && (
        <UploadModal onClose={() => setShowUpload(false)} onDone={handleUploadDone} />
      )}
    </div>
  );
}
