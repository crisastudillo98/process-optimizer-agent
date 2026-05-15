# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
make install                        # pip install -r requirements.txt

# Run development server (hot-reload, port 8000)
make dev                            # uvicorn api.main:app --reload

# Initialize RAG knowledge base (run once after first install)
python -m rag.seed_knowledge        # local dev
make seed                           # Docker

# Run tests
make test                           # unit tests, excludes test_api.py and integration tests
make test-api                       # FastAPI endpoint tests only
make test-all                       # all non-integration tests
make test-integration               # requires GROQ_API_KEY + running ChromaDB
make coverage                       # HTML report in htmlcov/

# Run a single test file
pytest tests/test_analyzer.py -v

# Run a single test function
pytest tests/test_analyzer.py::test_function_name -v

# Docker
make build && make up               # build and start all services
make logs                           # tail logs for process-optimizer container

# Cleanup
make clean                          # remove __pycache__, .pyc, pytest artifacts
make clean-all                      # also wipes storage/vector_db/
```

## Architecture

The system is a LangGraph-based agent that transforms natural language process descriptions (AS-IS) into optimized proposals (TO-BE) using Lean/Six Sigma/Kaizen, then generates BPMN 2.0 XML and quantitative KPIs.

### Pipeline (LangGraph StateGraph)

All pipeline state flows through a single `AgentState` Pydantic model (`models/schemas.py`). Each graph node receives the full state and returns a dict of fields to update.

```
load_document → extract_asis →[if ok]→ generate_asis_bpmn → analyze_waste
    → hitl_review_asis →[approved]→ retrieve_rag → optimize_tobe
    →[HITL_ENABLED]→ hitl_review →[approved/retries≥2]→ generate_bpmn → calculate_kpis
```

The graph is compiled once at import time as `optimizer_graph` in `agent/orchestrator.py`. Routing conditions are in the same file (`route_after_extraction`, `route_after_asis_hitl`, `route_after_optimization`, `route_after_hitl`).

**AS-IS HITL flow (Sprint 3):** after `analyze_waste`, the pipeline pauses at `hitl_review_asis`. The analyst reviews the extracted process via `GET /sessions/{id}/asis-review` and approves/rejects via `POST /sessions/{id}/asis-review`. If rejected with feedback, the pipeline re-runs `extract_asis` with the analyst's correction hint appended to `raw_input`. If approved (or no feedback), the pipeline continues to `retrieve_rag`.

**TO-BE HITL flow:** when `HITL_ENABLED=true`, the pipeline pauses at `hitl_review`. The `POST /sessions/{id}/review` endpoint injects the human decision into the in-memory `AgentState` and calls `_resume_pipeline` which re-invokes the graph with the same `thread_id`. Maximum 2 re-optimization retries before forcing continuation.

**HITL timeout:** an `asyncio` background task (`hitl_timeout_monitor`) runs on startup and checks every hour for sessions stalled in HITL for more than `HITL_TIMEOUT_HOURS` (default 24h). Timed-out sessions are marked `current_node="timed_out"`.

### Session storage

Active sessions are stored in `_sessions: dict[str, AgentState]` in `api/main.py` (in-process memory — not shared across workers). Completed analyses are persisted to SQLite via SQLAlchemy (`storage/`). **Session recovery (Sprint 3):** if a session is not in `_sessions` (e.g., after server restart), `_get_session()` checks SQLite via `repo.get_analysis()`. If found with `status="completed"`, it reconstructs a minimal `AgentState` via `repo.reconstruct_state_from_db()` and re-adds it to `_sessions`. The comment in the code flags this as "replace with Redis in production."

### LLM and embeddings

`llm/factory.py` is the single factory for all LLM and embedder instantiation. All agent nodes call `get_llm()`. Provider is configured via `LLM_PROVIDER` env var (`groq` default, `openai` supported, `perplexity` enum-only — not implemented). Embeddings use ChromaDB's local `DefaultEmbeddingFunction` (all-MiniLM-L6-v2, no API key needed).

### RAG

`rag/seed_knowledge.py` populates ChromaDB with Lean/Six Sigma patterns. `rag/retriever.py` exposes `node_retrieve_rag` which queries the vector store and injects results into `AgentState.rag_context` for the optimizer node.

### Prompts

Each agent node has a corresponding prompt module in `prompts/` (e.g., `prompts/extract_asis.py`, `prompts/detect_muda.py`). Prompts are versioned separately from the node logic.

### Key schemas (`models/schemas.py`)

- `AgentState` — the LangGraph shared state; all nodes read/write this
- `Process` — structured AS-IS process with `Activity` list
- `WasteAnalysisResult` — Lean analysis output (extends `WasteAnalysis`)
- `TOBEProcess` — optimized process with `OptimizedActivity` list
- `KPIReportV2` — 5 enriched KPIs + ROI + Sigma level (extends `KPIReport`)
- `BPMNOutput` — generated BPMN 2.0 XML + file path

### Frontend

The frontend is a multi-screen vanilla HTML/JS app with no build step. All files live in `docs/`. Shared CSS lives in `docs/css/main.css`; shared JS modules in `docs/js/`.

**Screens:**
- `docs/login.html` — Sign in / Create organization (auth entry point)
- `docs/dashboard.html` — Process list, search, new-process drawer
- `docs/workspace.html` — 3-panel: left nav, center chat, right living documentation (BPMN + KPIs + recommendations)
- `docs/metrics.html` — Full metrics view with AS-IS vs TO-BE comparison bars and Sigma level

**JS modules:**
- `docs/js/api.js` — All fetch calls. Exports `auth`, `analyses`, `sessions`, `chat`, `requireAuth`, `apiBlob`. Every request includes `Authorization: Bearer <token>` from `localStorage`. Redirects to `login.html` on 401.
- `docs/js/state.js` — In-memory state object (`user`, `currentSession`, `currentReport`, `processes`) shared across module imports within a page.

**Auth flow:** `login.html` → POST `/auth/login` or `/auth/register` → store tokens → `dashboard.html` → choose/create process → `workspace.html` → optionally `metrics.html`.

**localStorage keys:**
- `access_token` — JWT Bearer token for API calls
- `refresh_token` — stored for future refresh rotation
- `active_session` — current session/analysis ID passed between dashboard → workspace → metrics

`docs/app.html` is the legacy single-file UI, kept as a fallback (deprecated banner added).

## Environment setup

Copy `.env.example` to `.env`. Minimum required:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...         # free at console.groq.com
HITL_ENABLED=true            # set false to skip human review step
```

Database defaults to `sqlite:///storage/process_optimizer.db`. Set `DATABASE_URL` to a PostgreSQL connection string for production.

## Test markers

Tests are marked with `@pytest.mark.integration` when they require live API keys and ChromaDB. All other tests use mocked LLM calls via fixtures in `conftest.py`.

## Sprint 3 additions

### New AgentState fields (`models/schemas.py`)
- `hitl_asis_required: bool` — pipeline is paused at AS-IS HITL checkpoint
- `hitl_asis_approved: bool` — analyst has approved the AS-IS extraction
- `hitl_asis_feedback: str` — analyst's correction text (empty = no correction)
- `hitl_asis_started_at: Optional[datetime]` — when the HITL checkpoint was reached
- `bpmn_asis_output: Optional[str]` — file path to the generated AS-IS BPMN file

### New API endpoints
- `GET /sessions/{id}/asis-review` — returns `asis_process` dict + waste summary; requires `hitl_asis_required=True`
- `POST /sessions/{id}/asis-review` — body: `{approved: bool, feedback: str}`; resumes pipeline
- `GET /sessions/{id}/bpmn/asis` — returns AS-IS BPMN XML file download

### New repository helper (`storage/repository.py`)
- `reconstruct_state_from_db(record) -> dict` — builds minimal AgentState dict from SQLite `Analysis` record for session recovery

## Sprint 4 additions — Roles and Collaboration

### Business role concept
Two business roles are separate from the system role (owner/admin/member/viewer):
- `consultor` (default) — creates processes, invites collaborators, sees full pipeline, runs HITL reviews
- `colaborador` — invited to specific processes only; describes their part via chat; cannot create processes

`business_role` is stored in `users.business_role` and returned by `/auth/me`, `/auth/login`, `/auth/register`.

### New DB tables (`storage/models.py`)
- `process_collaborators` — links users to specific analyses as collaborators (status: pending|active|completed)
- `invitations` — external email invitations with a signed token (status: pending|accepted|expired, expires 7 days)
- `notifications` — in-app notifications (type: process_invitation|collaborator_completed|hitl_required|analysis_complete)

### New API endpoints (`api/main.py`)
**Collaboration:**
- `POST /analyses/{id}/invite` — invite by email; if user exists in tenant → creates `ProcessCollaborator` + `Notification`; if not → creates `Invitation` with token
- `GET /analyses/{id}/collaborators` — list collaborators with status (owner only)
- `POST /analyses/{id}/collaborators/{user_id}/complete` — collaborador marks their input done; notifies owner

**Notifications:**
- `GET /notifications?unread_only=true` — returns list + unread_count
- `POST /notifications/{id}/read` — mark one read
- `POST /notifications/read-all` — mark all read

**Invitations:**
- `POST /invitations/{token}/accept` — accept external process invite; creates user (business_role=colaborador) if new, creates `ProcessCollaborator` (active), issues JWT

### GET /analyses updated
Returns both owned analyses and analyses where user is a collaborator (status active/completed).
Each item now includes `is_owner: bool` and `collaborator_status: str | null`.

### auth/dependencies.py
Added `get_optional_user` — returns `User | None` without raising 401 (used by invitation acceptance).

### Frontend changes (`docs/`)
- `docs/js/api.js` — new `notifications` and `collaboration` export objects
- `docs/dashboard.html` — notification bell in topbar: unread badge, dropdown (last 5), polls every 30s
- `docs/workspace.html`:
  - Notification bell in topbar
  - TEAM section in left panel (consultors only): collaborator list + invite modal
  - Collaborador view (when `business_role === "colaborador"`): hides HITL banners, BPMN/metrics buttons; shows guided chat header with initial greeting; "Validate and Submit" button that calls complete endpoint

## Sprint 5 additions — Collaborative AS-IS

### Colaborador chat mode (`agent/chat_agent.py`)
`POST /chat` switches prompt based on `current_user.business_role`:
- `consultor` → existing `SYSTEM_PROMPT_TEMPLATE` (post-analysis Q&A)
- `colaborador` → `COLABORADOR_SYSTEM_PROMPT` (process elicitation: asks about activities, durations, tools, interactions, pain points, ideas)

The colaborador prompt receives `process_name` looked up from the analysis. `contexto_analisis` is allowed to be `{}` for colaboradores. Session IDs for collaborator chats follow `{analysis_id}_{user_id}_collab`, set by `docs/workspace.html` when the colaborador opens the workspace.

### ProcessCollaborator additions (`storage/models.py`)
- `session_id: str | None` — chat session for this collaborator (`{analysis_id}_{user_id}_collab`)
- `contribution_summary: text | null` — concatenated `Colaborador: …` / `Asistente: …` transcript persisted when they mark their work complete

Alembic migration: `f141c5557abc_sprint5_collaborator_chat`.

### New endpoints (`api/main.py`)
- `GET /analyses/{id}/collaborators/{user_id}/contribution` — owner-only; returns `{collaborator: {name, email, status, completed_at}, chat_history: [...], contribution_summary}`
- `POST /analyses/{id}/unify` — owner-only; requires ≥1 collaborator with `status="completed"`; calls the LLM with `UNIFY_SYSTEM_PROMPT` to synthesize a single AS-IS from all collaborator chats; writes the result into `Analysis.result_json["asis_process"]` and rebuilds `raw_input` from the joined chats; returns `{unified_asis, message}`. Returns 500 if the LLM fails or produces invalid JSON.

### Welcome notification
`POST /invitations/{token}/accept` now creates a `Notification(type="welcome")` only for the new-user branch (existing-user acceptance does not).

### Frontend (`docs/workspace.html`)
- TEAM section shows a "View" button next to each `status="completed"` collaborator → opens contribution modal with chat bubbles + completion metadata.
- Below the collaborators list, "⚡ Unify All Contributions" button (`.btn-unify`) appears only when at least one collaborator is completed. Calls `analyses.unify(sessionId)` and reloads the workspace on success.

## Sprint 6 additions — UI/UX + Branding

### Brand identity
- **Product name:** Processa AI (replaces ProcessOptix / ProcessOptimizer everywhere)
- **Tagline:** "Optimize your processes with collaborative AI"
- **Logo:** Connected-node SVG — three circles (left r=6, center r=9, right r=6) linked by two lines, all teal (`#00d4aa`). Inline SVG, no external file.

### Dark/Light theme toggle
- All pages load with theme from `localStorage.getItem('theme') || 'dark'` via an inline `<script>` in `<head>` (no flicker).
- `[data-theme="light"]` block in `docs/css/main.css` overrides `--bg`, `--sidebar-bg`, `--surface`, `--surface-2`, `--border`, `--border-med`, `--text`, `--muted`, `--very-muted`.
- Theme toggle button (`.theme-toggle-btn`) appears in every page topbar. Dark mode shows ☀️, light mode shows 🌙.
- `window.toggleTheme()` function defined in each page's module script.

### Personalized greeting (dashboard)
- After `GET /auth/me`, the dashboard title becomes a time-of-day greeting: "Buenos días/tardes/noches/Hola, {first_name} 👋".
- First name extracted from `user.full_name.split(' ')[0]`.
- Subtitle is role-specific: consultor → "Gestiona y optimiza…", colaborador → "Estos son los procesos en los que colaboras".

### Dashboard by role
**Consultor view** (unchanged UX, enhanced cards):
- Shows "+ New Process" and "Upload Sources" buttons.
- Shows "My Processes / Shared with me" tab switcher.
- Cards show collaborator count (`👥 N colaboradores`) when `collaborator_count > 0`.

**Colaborador view** (role = `colaborador`):
- Hides "+ New Process", "Upload Sources" buttons, and tab switcher.
- Shows `collab-banner` at top: "📋 Tienes X proceso(s) pendientes de tu contribución".
- Cards show contribution status badge: ⏳ Pendiente / 🔄 En progreso / ✅ Completado and owner name ("por {owner_name}").
- "Abrir Workspace" instead of "Open Workspace".

### CSS additions (`docs/css/main.css`)
- `.collab-banner` — teal-tinted info bar for colaboradores
- `.theme-toggle-btn` — small bordered button for theme switching

## Sprint 8 additions — Pipeline Redesign (collection-first + chat-driven HITL)

The pipeline now starts with **collection** rather than a single dumped AS-IS. The consultant creates the shell, collaborators describe their parts, and only then is the AS-IS unified, validated, optimized, and finalized. HITL no longer pauses inside the LangGraph orchestrator for the Sprint 8 flow — instead, each step is an explicit FastAPI endpoint that runs the relevant nodes directly and uses the `phase` column as the source of truth.

### Phase state machine (`analyses.phase`)
`collecting → unifying → asis_hitl → optimizing → tobe_hitl → completed`

The `Analysis` table now has two new columns:
- `department: String(255)` — organizational area, captured at creation time.
- `phase: String(50)` — current state in the machine above.

`ProcessCollaborator.rag_indexed: Boolean` tracks whether the collaborator's uploaded documents have been processed into tenant RAG.

Alembic migration: `3ed4d2fd9aeb_sprint8_pipeline_redesign`. Backfills `phase='completed'` for legacy rows where `status='completed'` so they keep rendering as final reports.

### New endpoints (`api/main.py`)
All gated to the process owner and to a specific phase (returns 409 otherwise):

- `POST /processes` — body `{process_name, department, description}`. Creates the row in `phase="collecting"`. **No pipeline runs.** Returns `session_id`.
- `POST /processes/{id}/start-unification` — requires ≥1 completed collaborator. Sets `phase="unifying"` and runs `_run_unification()` in a background task.
- `POST /processes/{id}/approve-asis` — phase=`asis_hitl` → `optimizing`. Runs `_run_optimization()` (waste analysis → RAG → optimize_tobe) in background.
- `POST /processes/{id}/approve-tobe` — phase=`tobe_hitl`. Runs `_generate_final_report()` (AS-IS BPMN + TO-BE BPMN + KPIs + Muda/Mura/Muri + tool recs) in background; the task flips phase to `completed` when done.
- `POST /processes/{id}/request-revision` — body `{phase, feedback}`. Saves feedback as a user chat message and re-runs unification or optimization with the feedback as a correction hint.

### Background tasks
- `_run_unification(analysis_id, owner_id, feedback="")` — pulls all completed collaborator chats + consultant's initial description, calls the LLM with `SPRINT8_UNIFY_PROMPT`, builds a real `Process` pydantic via `_build_process_from_unified()`, persists to `result_json["asis_process"]` and to `_sessions`, posts `_format_asis_for_chat()` into the consultant's chat, sets `phase="asis_hitl"`.
- `_run_optimization(analysis_id, feedback="")` — recovers AS-IS from memory or DB, calls `node_analyze_waste` → `node_retrieve_rag` → `node_optimize_tobe` sequentially, posts `_format_tobe_for_chat()` into chat, sets `phase="tobe_hitl"`.
- `_generate_final_report(analysis_id)` — calls `node_generate_asis_bpmn` → `node_generate_bpmn` → `node_calculate_kpis`, runs `build_enrichment_block()` (Muda/Mura/Muri + tool recs), saves everything via `repo.complete_analysis()` + `repo.update_bpmn_paths()`, sets `phase="completed"`, posts the "🎉 Reporte final generado!" message.

### Collaborator chat — dynamic prompt
`agent/chat_agent.build_colaborador_prompt(process_name, department, collaborator_name, conversation_history)` replaces the static `COLABORADOR_SYSTEM_PROMPT`. It calls `_analyze_collected_info(history)` which uses Spanish keyword bags (`_TIME_HINTS`, `_TOOL_HINTS`, `_PAIN_HINTS`, `_PEOPLE_HINTS`) to detect which of {activities, durations, tools, pain_points, people} have already been mentioned and injects an "INFORMACIÓN AÚN POR PROFUNDIZAR" line so the LLM probes only the gaps. Used by `/chat` whenever `current_user.business_role == "colaborador"`. The legacy `COLABORADOR_SYSTEM_PROMPT` constant is kept as a tiny fallback for any direct import.

### Collaborator document processing
`POST /sessions/{session_id}/sources` is now role-aware:
- **Colaborador** branch — `session_id` is normalized to `analysis_id`, the user must be a `ProcessCollaborator` on it. The file is stored under `storage/outputs/sources/{analysis_id}/colab/{user_id}/`, the LLM extracts a focused 5-section Spanish summary via `COLAB_DOC_EXTRACTION_PROMPT` (activities / durations / tools / people / pain points), and the result is posted as an `assistant` message into `{analysis_id}_{user_id}_collab` so the agent can react to it. `ProcessCollaborator.rag_indexed` is set to True.
- **Consultor / owner** branch — unchanged (appends to `AgentState.rag_context`).

### HITL is no longer banner-based
The yellow `.hitl-banner` is hidden globally via CSS (`display: none !important`). Sprint 8 validation lives in a new left-panel `#validation-section` (`docs/workspace.html`):
- `#asis-validation` with [✓ Aprobar AS-IS] [✗ Solicitar cambios]
- `#tobe-validation` with [✓ Aprobar TO-BE] [✗ Solicitar cambios]

When the consultant clicks "Solicitar cambios", a `window.__pendingRevision` flag is set; the next chat message is routed through `processes.requestRevision()` instead of `/chat`. The agent regenerates and posts the new AS-IS / TO-BE to chat.

The workspace polls via `pollStatus()`:
- If `status.phase` is set (Sprint 8 row) → `handlePhaseUpdate(phase)` flips the status label and shows the matching validation block.
- Else → legacy state-flag fallback for old `/analyze/text` rows.

`maybeRefreshChat()` re-fetches chat history each poll tick so AS-IS / TO-BE summary bubbles posted by background tasks appear without a full page reload.

### Enrichment block (Muda / Mura / Muri + tool recommendations)
`agent/waste_enrichment.py` — pure functions over existing schemas:
- `classify_muda_mura_muri(asis, waste)` returns three lists. Mura = activities ≥ 1σ off the duration mean (when ≥3 activities exist). Muri = activities ≥ 2× the process average duration that are owned by a single role.
- `recommend_tools(tobe, waste)` matches TO-BE activities (status ∈ automatizada / optimizada / combinada) to entries in `TOOL_CATALOG` (7 categories — document_processing, approval_workflow, analytics_dashboards, communication_orchestration, data_capture_forms, rpa_attended, queue_optimization) via waste-type triggers + keyword matching. Each entry has tool names, cost range in USD/mes, ROI months, and the embodied methodology.

The block is stored under `result_json["enrichment"]` and surfaced by `GET /sessions/{id}/report`. The Recommendations tab in the workspace renders it as cards.

### Frontend API (`docs/js/api.js`)
New `processes` export with `create`, `startUnification`, `approveAsis`, `approveTobe`, `requestRevision`.

### Dashboard cards
Consultor cards now show phase-specific status pills: 📥 Recolectando / ⚡ Unificando / ⏸ Validar AS-IS / 🧠 Optimizando / ⏸ Validar TO-BE / Completed. Department badge shown under the date. Falls back to legacy `status` when `phase` is null.

### Backward compatibility
- `POST /analyze/text` still works — runs the full LangGraph pipeline as before, on rows with `phase=NULL`. The endpoint is marked `[Legacy]` in its OpenAPI summary.
- `POST /analyses/{id}/unify` (Sprint 5) is preserved as a blocking, synchronous alternative to the new `POST /processes/{id}/start-unification`. New clients should use the latter.
- Legacy HITL endpoints (`/sessions/{id}/review`, `/sessions/{id}/asis-review`) still exist but the yellow banner UI is gone; legacy HITL approvals would need to be issued via the API directly.

## Known limitations

- Sessions are in-process memory; restarting the server loses **active** (non-Sprint-8) AgentStates. Sprint-8 rows can be reconstructed from `phase` + `result_json` and are recovered on demand, but the legacy `/analyze/text` flow still loses live LangGraph state on restart.
- Legacy HITL endpoints (`/sessions/{id}/review`, `/sessions/{id}/asis-review`) keep working at the API level but have no banner UI anymore — Sprint 8 deliberately removed the yellow HITL banners.
- `perplexity` is present in `LLM_PROVIDER` enum but has no implementation — do not use.
- VAR (Value-Added Ratio) can report 100% when all AS-IS activities are declared value-added.
- `ProcessCollaborator.rag_indexed=True` currently signals intent — collaborator documents are summarized and posted to chat, but they are not yet pushed into the tenant-scoped vector store.
