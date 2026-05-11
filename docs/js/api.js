const API_BASE = 'http://localhost:8000';

function getToken() {
  return localStorage.getItem('access_token');
}

function authHeaders() {
  const token = getToken();
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
  };
}

export function requireAuth() {
  if (!getToken()) {
    window.location.href = 'login.html';
    return false;
  }
  return true;
}

async function apiRequest(method, path, body = null) {
  const opts = { method, headers: authHeaders() };
  if (body) opts.body = JSON.stringify(body);

  const res = await fetch(`${API_BASE}${path}`, opts);

  if (res.status === 401) {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    window.location.href = 'login.html';
    return null;
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

async function apiUpload(path, formData) {
  const token = getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: token ? { 'Authorization': `Bearer ${token}` } : {},
    body: formData
  });
  if (res.status === 401) {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    window.location.href = 'login.html';
    return null;
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function apiBlob(path) {
  const token = getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    headers: token ? { 'Authorization': `Bearer ${token}` } : {}
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.blob();
}

// Auth
export const auth = {
  login: (email, password) =>
    apiRequest('POST', '/auth/login', { email, password }),
  register: (data) =>
    apiRequest('POST', '/auth/register', data),
  me: () =>
    apiRequest('GET', '/auth/me'),
  logout: () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('active_session');
    window.location.href = 'login.html';
  }
};

// Analyses (persisted, paginated list)
export const analyses = {
  list: (limit = 50) =>
    apiRequest('GET', `/analyses?limit=${limit}`),
  get: (id) =>
    apiRequest('GET', `/analyses/${id}`),
  getContribution: (analysisId, userId) =>
    apiRequest('GET', `/analyses/${analysisId}/collaborators/${userId}/contribution`)
};

// Sessions (live pipeline sessions)
export const sessions = {
  analyzeText: (raw_input, process_name) =>
    apiRequest('POST', '/analyze/text', { raw_input, process_name }),
  analyzeFile: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return apiUpload('/analyze/file', fd);
  },
  status: (id) =>
    apiRequest('GET', `/sessions/${id}/status`),
  process: (id) =>
    apiRequest('GET', `/sessions/${id}/process`),
  report: (id) =>
    apiRequest('GET', `/sessions/${id}/report`),
  kpis: (id) =>
    apiRequest('GET', `/sessions/${id}/kpis`),
  bpmn: async (id, filename = 'process.bpmn') => {
    const blob = await apiBlob(`/sessions/${id}/bpmn`);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  },
  review: (id, approved, feedback = '') =>
    apiRequest('POST', `/sessions/${id}/review`, { approved, feedback }),
  asisReview: (id) =>
    apiRequest('GET', `/sessions/${id}/asis-review`),
  asisApprove: (id, approved, feedback = '') =>
    apiRequest('POST', `/sessions/${id}/asis-review`, { approved, feedback }),
  bpmnAsis: async (id, filename = 'process_asis.bpmn') => {
    const blob = await apiBlob(`/sessions/${id}/bpmn/asis`);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  },
  delete: (id) =>
    apiRequest('DELETE', `/sessions/${id}`)
};

// Sources
export const sources = {
  upload: (sessionId, file) => {
    const fd = new FormData();
    fd.append('file', file);
    return apiUpload(`/sessions/${sessionId}/sources`, fd);
  },
  list: (sessionId) =>
    apiRequest('GET', `/sessions/${sessionId}/sources`)
};

// Refinement
export const refinement = {
  refine: (sessionId, instruction) =>
    apiRequest('POST', `/sessions/${sessionId}/refine`, { instruction })
};

// Chat
export const chat = {
  send: (session_id, mensaje, contexto_analisis) =>
    apiRequest('POST', '/chat', { session_id, mensaje, contexto_analisis }),
  history: (id) =>
    apiRequest('GET', `/chat/${id}/history`)
};

// Notifications
export const notifications = {
  list: (unread_only = false) =>
    apiRequest('GET', `/notifications?unread_only=${unread_only}`),
  markRead: (id) =>
    apiRequest('POST', `/notifications/${id}/read`),
  markAllRead: () =>
    apiRequest('POST', '/notifications/read-all')
};

// Collaboration
export const collaboration = {
  invite: (analysisId, email, message = '') =>
    apiRequest('POST', `/analyses/${analysisId}/invite`, { email, message }),
  listCollaborators: (analysisId) =>
    apiRequest('GET', `/analyses/${analysisId}/collaborators`),
  complete: (analysisId, userId) =>
    apiRequest('POST', `/analyses/${analysisId}/collaborators/${userId}/complete`),
  getInvitation: (token) =>
    apiRequest('GET', `/invitations/${token}`),
  acceptInvite: (token, data) =>
    apiRequest('POST', `/invitations/${token}/accept`, data)
};

export { getToken };
