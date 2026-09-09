// Single source of truth for the API base URL.
// Set VITE_API_BASE_URL in frontend/.env to override.
export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
  // evidence-image returns binary
  if (options._raw) return res;
  return res.json();
}

export const api = {
  // GET /documents → returns { documents: [...], total_facts, relationships_by_type }
  getDocuments: () => request('/documents').then(r => r.documents ?? r),

  // POST /documents (multipart)
  uploadDocument: (formData) =>
    request('/documents', { method: 'POST', body: formData }),

  // GET /documents/{id}/status
  getDocumentStatus: (id) => request(`/documents/${id}/status`),

  // GET /facts?document_id=&search=&limit=&offset= → returns { total, data: [...] }
  getFacts: (params = {}) => {
    const q = new URLSearchParams();
    if (params.document_id) q.set('document_id', params.document_id);
    if (params.search) q.set('search', params.search);
    if (params.limit) q.set('limit', params.limit);
    if (params.offset) q.set('offset', params.offset);
    const qs = q.toString() ? `?${q}` : '';
    return request(`/facts${qs}`).then(r => r.data ?? r);
  },

  // GET /facts/{id}
  getFact: (id) => request(`/facts/${id}`),

  // GET /facts/{id}/relationships
  getRelationships: (id) => request(`/facts/${id}/relationships`),

  // GET /facts/{id}/evidence-image  — returns a URL we can use as <img src>
  getEvidenceImageUrl: (id) => `${API_BASE}/facts/${id}/evidence-image`,
};
