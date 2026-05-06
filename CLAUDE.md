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
load_document → extract_asis →[if ok]→ analyze_waste → retrieve_rag → optimize_tobe
    →[HITL_ENABLED]→ hitl_review →[approved/retries≥2]→ generate_bpmn → calculate_kpis
```

The graph is compiled once at import time as `optimizer_graph` in `agent/orchestrator.py`. Routing conditions are in the same file (`route_after_extraction`, `route_after_optimization`, `route_after_hitl`).

**HITL flow:** when `HITL_ENABLED=true`, the pipeline pauses at `hitl_review`. The `POST /sessions/{id}/review` endpoint injects the human decision into the in-memory `AgentState` and calls `_resume_pipeline` which re-invokes the graph with the same `thread_id`. Maximum 2 re-optimization retries before forcing continuation.

### Session storage

Active sessions are stored in `_sessions: dict[str, AgentState]` in `api/main.py` (in-process memory — not shared across workers). Completed analyses are persisted to SQLite via SQLAlchemy (`storage/`). The comment in the code flags this as "replace with Redis in production."

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

`docs/app.html` is a standalone single-file web UI (HTML + CSS + JS, no build step). It calls the FastAPI backend and renders results including a dynamic BPMN diagram.

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

## Known limitations

- Sessions are in-process memory; restarting the server loses active (non-persisted) sessions.
- HITL sessions with no `/review` call will remain paused indefinitely (no timeout).
- `perplexity` is present in `LLM_PROVIDER` enum but has no implementation — do not use.
- VAR (Value-Added Ratio) can report 100% when all AS-IS activities are declared value-added.
