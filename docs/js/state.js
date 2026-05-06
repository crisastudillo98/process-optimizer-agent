export const state = {
  user: null,           // from GET /auth/me
  currentSession: null, // active session id
  currentReport: null,  // full report object
  processes: [],        // list from GET /analyses
};

export function setUser(user) { state.user = user; }
export function setSession(id) { state.currentSession = id; }
export function setReport(report) { state.currentReport = report; }
export function setProcesses(list) { state.processes = list; }
