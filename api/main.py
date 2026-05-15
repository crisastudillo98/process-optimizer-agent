from __future__ import annotations
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, BackgroundTasks, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.orchestrator import optimizer_graph, build_graph
from agent.document_loader import load_document
from agent.chat_agent import router as chat_router
from api.auth import router as auth_router
from auth.dependencies import get_current_user, get_optional_user, require_admin
from models.schemas import (
    AgentState,
    KPIReportV2,
    TOBEProcess,
    WasteAnalysisResult,
    BPMNOutput,
)
from config.settings import settings
from observability.logger import get_logger
from storage.database import SessionLocal
from storage import models as db_models
from storage.models import User, ProcessCollaborator, Invitation, Notification
from storage import repository as repo
# Schema is managed by Alembic — do NOT call create_all here

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Process Optimizer Agent",
    description=(
        "Agente inteligente para optimización de procesos empresariales. "
        "Transforma descripciones AS-IS en propuestas TO-BE optimizadas "
        "con Lean, Six Sigma y Kaizen."
    ),
    version="1.0.0",
    docs_url="/openapi",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(auth_router, prefix="/auth", tags=["Auth"])

# ─────────────────────────────────────────────
# STATIC FILES + ROOT REDIRECT
# ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("docs/login.html")

app.mount("/docs", StaticFiles(directory="docs"), name="docs")

# ─────────────────────────────────────────────
# SESSION STORE  (in-memory; replace with Redis in prod)
# ─────────────────────────────────────────────

_sessions: dict[str, AgentState] = {}
_recovered_sessions: set[str] = set()

# Source upload metadata (in-memory; replace with DB in production)
_session_sources: dict[str, list[dict]] = {}


def _get_session(session_id: str, user_id: Optional[str] = None) -> AgentState:
    if session_id in _sessions:
        return _sessions[session_id]

    # Session not in memory — try to recover from SQLite
    with SessionLocal() as db:
        record = repo.get_analysis(db, session_id, user_id=user_id)
    if record and record.status == "completed":
        state_dict = repo.reconstruct_state_from_db(record)
        try:
            state = AgentState(**state_dict)
        except Exception:
            state = AgentState(
                raw_input=record.raw_input or "",
                current_node="calculate_kpis",
                kpi_ok=True,
                user_id=record.user_id,
                tenant_id=record.tenant_id,
            )
        _sessions[session_id] = state
        _recovered_sessions.add(session_id)
        logger.info(f"Sesión {session_id} recuperada de SQLite")
        return state

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Sesión '{session_id}' no encontrada.",
    )


def _assert_session_owner(state: AgentState, current_user: User) -> None:
    """Raises 403 if the session belongs to a different user."""
    if state.user_id and state.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to this session.")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class AnalyzeTextRequest(BaseModel):
    raw_input: str = Field(
        ...,
        min_length=50,
        description="Descripción del proceso AS-IS en lenguaje natural (mínimo 50 chars)",
        examples=[
            "El proceso de facturación inicia cuando el cliente aprueba el pedido. "
            "Un asistente revisa los datos (30 min), genera la factura en SAP (20 min), "
            "envía por correo (5 min) y espera confirmación hasta 2 días hábiles."
        ],
    )
    process_name: str = Field(
        default="Sin nombre",
        max_length=255,
        description="Nombre descriptivo del análisis (ej: Proceso de contratación RRHH)",
    )


class CreateProcessRequest(BaseModel):
    """Sprint 8 — collection-first process creation (no immediate pipeline)."""
    process_name: str = Field(..., min_length=2, max_length=255)
    department:   str = Field(..., min_length=2, max_length=255)
    description:  str = Field(default="", max_length=2000)


class RevisionRequest(BaseModel):
    phase:    str = Field(..., description="asis_hitl | tobe_hitl")
    feedback: str = Field(..., min_length=1)


class HITLReviewRequest(BaseModel):
    approved: bool = Field(..., description="True = aprobado, False = requiere cambios")
    feedback: str  = Field(default="", description="Comentarios del revisor")


class RefineRequest(BaseModel):
    instruction: str = Field(..., min_length=5, description="Natural-language instruction to modify the TO-BE process")


class SessionResponse(BaseModel):
    session_id:   str
    message:      str
    current_node: str
    status:       str


class AnalysisStatusResponse(BaseModel):
    session_id:          str
    current_node:        str
    extraction_ok:       bool
    analysis_ok:         bool
    optimization_ok:     bool
    hitl_asis_required:  bool
    hitl_asis_approved:  bool
    hitl_required:       bool
    hitl_approved:       bool
    bpmn_ok:             bool
    kpi_ok:              bool
    errors:              list[str]
    recovered_from_db:   bool = False
    process_name:        Optional[str] = None
    # Sprint 8 — phase state machine
    phase:               Optional[str] = None
    department:          Optional[str] = None


# ─────────────────────────────────────────────
# HEALTH  (public)
# ─────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
async def health_check():
    return {
        "status":  "healthy",
        "version": "1.0.0",
        "hitl":    settings.hitl_enabled,
        "model":   settings.openai_model,
    }


@app.get("/health/rag", tags=["Sistema"])
async def health_rag():
    try:
        from rag.vector_store import get_collection_stats
        stats = get_collection_stats()
        return {"status": "healthy", "collections": stats}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector DB no disponible: {str(e)}",
        )


# ─────────────────────────────────────────────
# SPRINT 8 — PROCESS CREATION (collection-first)
# ─────────────────────────────────────────────

@app.post("/processes", tags=["Processes"], status_code=status.HTTP_201_CREATED)
async def create_process(
    req: CreateProcessRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Create a new process in 'collecting' phase. No pipeline runs yet.
    The consultant invites collaborators, who describe their parts via chat.
    Once at least one collaborator completes, the consultant triggers /start-unification.
    """
    analysis_id = str(uuid.uuid4())
    with SessionLocal() as db:
        record = db_models.Analysis(
            id=analysis_id,
            process_name=req.process_name,
            department=req.department,
            raw_input=req.description or "",
            status="running",      # status remains "running" until phase=="completed"
            phase="collecting",
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
        )
        db.add(record)
        db.commit()

    state = AgentState(
        raw_input=req.description or "",
        current_node="collecting",
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    _sessions[analysis_id] = state
    logger.info(f"Sprint8 process created: {analysis_id} name={req.process_name} dept={req.department}")

    return {
        "session_id":   analysis_id,
        "status":       "collecting",
        "phase":        "collecting",
        "process_name": req.process_name,
        "department":   req.department,
    }


# ─────────────────────────────────────────────
# ANÁLISIS — TEXTO LIBRE
# ─────────────────────────────────────────────

@app.post(
    "/analyze/text",
    response_model=SessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Análisis"],
    summary="[Legacy] Analizar proceso desde texto libre — runs full LangGraph pipeline",
    description=(
        "Legacy endpoint kept for backward compatibility. New clients should use "
        "POST /processes (Sprint 8 collection-first flow) which lets collaborators "
        "describe their parts before unification + optimization."
    ),
)
async def analyze_text(
    request: AnalyzeTextRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    session_id = str(uuid.uuid4())
    state = AgentState(
        raw_input=request.raw_input,
        current_node="start",
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    _sessions[session_id] = state

    with SessionLocal() as db:
        repo.create_analysis(
            db, session_id, request.process_name, request.raw_input,
            user_id=current_user.id, tenant_id=current_user.tenant_id,
        )

    logger.info(f"Nueva sesión: {session_id} user={current_user.id} chars={len(request.raw_input)}")
    background_tasks.add_task(_run_pipeline, session_id, request.process_name)

    return SessionResponse(
        session_id=session_id,
        message="Análisis iniciado. Consulta /sessions/{session_id}/status para ver el progreso.",
        current_node="start",
        status="running",
    )


# ─────────────────────────────────────────────
# ANÁLISIS — ARCHIVO
# ─────────────────────────────────────────────

@app.post(
    "/analyze/file",
    response_model=SessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Análisis"],
    summary="Analizar proceso desde archivo (PDF, Excel, TXT)",
)
async def analyze_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    allowed_extensions = {".pdf", ".xlsx", ".xls", ".txt", ".md", ".json"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Extensión '{suffix}' no soportada. "
                f"Extensiones válidas: {sorted(allowed_extensions)}"
            ),
        )

    tmp_dir  = Path("/tmp/process_optimizer")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    tmp_path.write_bytes(await file.read())

    try:
        raw_text = load_document(tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No se pudo leer el archivo: {str(e)}",
        )

    if len(raw_text.strip()) < 50:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El archivo tiene contenido insuficiente (mínimo 50 caracteres).",
        )

    session_id = str(uuid.uuid4())
    state = AgentState(
        raw_input=raw_text,
        input_file_path=str(tmp_path),
        current_node="start",
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    _sessions[session_id] = state

    with SessionLocal() as db:
        repo.create_analysis(
            db, session_id, file.filename or "uploaded_file", raw_text,
            user_id=current_user.id, tenant_id=current_user.tenant_id,
        )

    logger.info(f"Nueva sesión desde archivo: {session_id} {file.filename} ({len(raw_text)} chars)")
    background_tasks.add_task(_run_pipeline, session_id)

    return SessionResponse(
        session_id=session_id,
        message=f"Archivo '{file.filename}' recibido. Análisis iniciado.",
        current_node="start",
        status="running",
    )


# ─────────────────────────────────────────────
# PIPELINE RUNNER (background task)
# ─────────────────────────────────────────────

async def _run_pipeline(session_id: str, process_name: str = "Sin nombre") -> None:
    if session_id not in _sessions:
        logger.error(f"Pipeline abortado: sesión {session_id} no existe en memoria")
        return

    state = _sessions[session_id]
    try:
        logger.info(f"Pipeline iniciado: {session_id}")
        result = optimizer_graph.invoke(
            state.model_dump(),
            config={"configurable": {"thread_id": session_id}},
        )

        if result and isinstance(result, dict):
            _sessions[session_id] = AgentState(**result)
            final_state = _sessions[session_id]

            # Pipeline paused at AS-IS HITL — do not mark as complete
            if final_state.hitl_asis_required and not final_state.hitl_asis_approved:
                logger.info(f"Pipeline pausado en AS-IS HITL checkpoint: {session_id}")
                return

            full_report = {
                "asis_process":   final_state.asis_process.model_dump() if final_state.asis_process else None,
                "waste_analysis": final_state.waste_analysis.model_dump() if final_state.waste_analysis else None,
                "tobe_process":   final_state.tobe_process.model_dump() if final_state.tobe_process else None,
                "kpi_report":     final_state.kpi_report.model_dump() if final_state.kpi_report else None,
            }
            score = final_state.waste_analysis.waste_percentage if final_state.waste_analysis else None
            with SessionLocal() as db:
                repo.complete_analysis(db, session_id, full_report, score)
            bpmn_tobe = final_state.bpmn_output.file_path if final_state.bpmn_output else None
            bpmn_asis = final_state.bpmn_asis_output
            if bpmn_tobe or bpmn_asis:
                with SessionLocal() as db:
                    repo.update_bpmn_paths(db, session_id, bpmn_tobe, bpmn_asis)
        else:
            _sessions[session_id].current_node = "error"
            _sessions[session_id].errors.append("pipeline: resultado inválido del grafo")
            with SessionLocal() as db:
                repo.fail_analysis(db, session_id, _sessions[session_id].errors)

        logger.info(f"Pipeline completado: {session_id} nodo={_sessions[session_id].current_node}")

    except Exception as e:
        logger.error(f"Error en pipeline {session_id}: {e}")
        if session_id in _sessions:
            _sessions[session_id].errors.append(f"pipeline: {str(e)}")
            _sessions[session_id].current_node = "error"
            with SessionLocal() as db:
                repo.fail_analysis(db, session_id, _sessions[session_id].errors)
        else:
            logger.error(f"Sesión {session_id} desapareció durante el pipeline")


# ─────────────────────────────────────────────
# SESIONES — STATUS Y RESULTADOS
# ─────────────────────────────────────────────

@app.get(
    "/sessions/{session_id}/status",
    response_model=AnalysisStatusResponse,
    tags=["Sesiones"],
)
async def get_session_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)

    recovered = session_id in _recovered_sessions
    process_name: Optional[str] = None
    phase: Optional[str] = None
    department: Optional[str] = None
    with SessionLocal() as db:
        record = repo.get_analysis(db, session_id, user_id=current_user.id)
    if record:
        process_name = record.process_name
        phase = record.phase
        department = record.department

    return AnalysisStatusResponse(
        session_id=session_id,
        current_node=state.current_node,
        extraction_ok=state.extraction_ok,
        analysis_ok=state.analysis_ok,
        optimization_ok=state.optimization_ok,
        hitl_asis_required=state.hitl_asis_required,
        hitl_asis_approved=state.hitl_asis_approved,
        hitl_required=state.hitl_required,
        hitl_approved=state.hitl_approved,
        bpmn_ok=state.bpmn_ok,
        kpi_ok=state.kpi_ok,
        errors=state.errors,
        recovered_from_db=recovered,
        process_name=process_name,
        phase=phase,
        department=department,
    )


@app.get("/sessions/{session_id}/process", tags=["Resultados"])
async def get_asis_process(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id)
    _assert_session_owner(state, current_user)
    if not state.extraction_ok or state.asis_process is None:
        raise HTTPException(status_code=425, detail="El proceso AS-IS aún no está disponible.")
    return state.asis_process.model_dump()


@app.get("/sessions/{session_id}/analysis", tags=["Resultados"])
async def get_waste_analysis(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id)
    _assert_session_owner(state, current_user)
    if not state.analysis_ok or state.waste_analysis is None:
        raise HTTPException(status_code=425, detail="El análisis Lean aún no está disponible.")
    return state.waste_analysis.model_dump()


@app.get("/sessions/{session_id}/tobe", tags=["Resultados"])
async def get_tobe_process(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id)
    _assert_session_owner(state, current_user)
    if not state.optimization_ok or state.tobe_process is None:
        raise HTTPException(status_code=425, detail="La propuesta TO-BE aún no está disponible.")
    return state.tobe_process.model_dump()


@app.get("/sessions/{session_id}/kpis", tags=["Resultados"])
async def get_kpi_report(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id)
    _assert_session_owner(state, current_user)
    if not state.kpi_ok or state.kpi_report is None:
        raise HTTPException(status_code=425, detail="El reporte de KPIs aún no está disponible.")
    return state.kpi_report.model_dump()


@app.get("/sessions/{session_id}/bpmn", tags=["Resultados"], response_class=FileResponse)
async def download_bpmn(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)

    # Try in-memory state first
    if state.bpmn_ok and state.bpmn_output:
        file_path = Path(state.bpmn_output.file_path)
        if file_path.exists():
            return FileResponse(path=str(file_path), media_type="application/xml", filename=file_path.name)

    # Fall back to DB-persisted path (recovered sessions / restart scenario)
    with SessionLocal() as db:
        record = repo.get_analysis(db, session_id, user_id=current_user.id)
    if record and record.bpmn_tobe_path:
        file_path = Path(record.bpmn_tobe_path)
        if file_path.exists():
            return FileResponse(path=str(file_path), media_type="application/xml", filename=file_path.name)

    # Completed sessions with no BPMN path are old sessions — return 404, not 425
    if record and record.status == "completed":
        raise HTTPException(status_code=404, detail="BPMN not available for this session")

    raise HTTPException(status_code=425, detail="El diagrama BPMN aún no está disponible.")


@app.get("/sessions/{session_id}/bpmn/asis", tags=["Resultados"], response_class=FileResponse)
async def download_asis_bpmn(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)

    # Try in-memory state first
    if state.bpmn_asis_output:
        file_path = Path(state.bpmn_asis_output)
        if file_path.exists():
            return FileResponse(path=str(file_path), media_type="application/xml", filename=file_path.name)

    # Fall back to DB-persisted path (recovered sessions / restart scenario)
    with SessionLocal() as db:
        record = repo.get_analysis(db, session_id, user_id=current_user.id)
    if record and record.bpmn_asis_path:
        file_path = Path(record.bpmn_asis_path)
        if file_path.exists():
            return FileResponse(path=str(file_path), media_type="application/xml", filename=file_path.name)

    raise HTTPException(status_code=425, detail="El diagrama BPMN AS-IS aún no está disponible.")


@app.get("/sessions/{session_id}/report", tags=["Resultados"])
async def get_full_report(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)
    if not state.kpi_ok:
        raise HTTPException(
            status_code=425,
            detail="El análisis completo aún no está disponible.",
        )

    # Sprint 8 enrichment — fetch any stored enrichment block (Muda/Mura/Muri + tools)
    enrichment_block: Optional[dict] = None
    with SessionLocal() as db:
        record_for_enrich = repo.get_analysis(db, session_id, user_id=current_user.id)
    if record_for_enrich and record_for_enrich.result_json:
        try:
            enrichment_block = json.loads(record_for_enrich.result_json).get("enrichment")
        except (json.JSONDecodeError, AttributeError):
            enrichment_block = None

    # State fields may be absent when recovered with the minimal fallback — read directly from SQLite
    if state.asis_process is None:
        with SessionLocal() as db:
            record = repo.get_analysis(db, session_id, user_id=current_user.id)
        if record and record.result_json:
            result = json.loads(record.result_json)
            return {
                "session_id":     session_id,
                "asis_process":   result.get("asis_process"),
                "waste_analysis": result.get("waste_analysis"),
                "tobe_process":   result.get("tobe_process"),
                "kpi_report":     result.get("kpi_report"),
                "enrichment":     result.get("enrichment"),
                "bpmn_file":      None,
            }

    return {
        "session_id":     session_id,
        "asis_process":   state.asis_process.model_dump() if state.asis_process else None,
        "waste_analysis": state.waste_analysis.model_dump() if state.waste_analysis else None,
        "tobe_process":   state.tobe_process.model_dump() if state.tobe_process else None,
        "kpi_report":     state.kpi_report.model_dump() if state.kpi_report else None,
        "enrichment":     enrichment_block,
        "bpmn_file":      state.bpmn_output.file_path if state.bpmn_output else None,
    }


# ─────────────────────────────────────────────
# HITL — REVISIÓN HUMANA
# ─────────────────────────────────────────────

@app.post("/sessions/{session_id}/review", response_model=SessionResponse, tags=["HITL"])
async def submit_hitl_review(
    session_id: str,
    review: HITLReviewRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id)
    _assert_session_owner(state, current_user)

    if not state.hitl_required:
        raise HTTPException(status_code=400, detail="Esta sesión no tiene una revisión humana pendiente.")

    if not review.approved and not review.feedback.strip():
        raise HTTPException(
            status_code=422,
            detail="Si el TO-BE es rechazado, debes proporcionar feedback para re-optimizar.",
        )

    state.hitl_approved = review.approved
    state.hitl_feedback = review.feedback or None
    state.hitl_required = False

    if review.approved and state.tobe_process:
        state.tobe_process.human_approved = True
        state.tobe_process.approver_notes = review.feedback or None

    _sessions[session_id] = state
    background_tasks.add_task(_resume_pipeline, session_id)

    action = "aprobado" if review.approved else "rechazado — re-optimizando"
    logger.info(f"HITL {session_id}: {action}")

    return SessionResponse(
        session_id=session_id,
        message=f"Revisión registrada: TO-BE {action}.",
        current_node=state.current_node,
        status="running",
    )


async def _resume_pipeline(session_id: str) -> None:
    from rag.retriever import node_retrieve_rag
    from agent.optimizer import node_optimize_tobe, node_hitl_review
    from agent.bpmn_generator import node_generate_bpmn
    from agent.kpi_calculator import node_calculate_kpis

    state = _sessions[session_id]
    try:
        # AS-IS HITL resume: call remaining nodes directly instead of
        # re-invoking the full graph (which always starts from load_document)
        if state.hitl_asis_approved and not state.optimization_ok:
            logger.info(f"Reanudando pipeline post-AS-IS HITL: {session_id}")
            current = state.model_dump()
            for node_fn in [
                node_retrieve_rag,
                node_optimize_tobe,
                node_hitl_review,
                node_generate_bpmn,
                node_calculate_kpis,
            ]:
                update = node_fn(AgentState(**current))
                current.update(update)
                _sessions[session_id] = AgentState(**current)
                # Pause here if TO-BE HITL is triggered
                if current.get("hitl_required") and not current.get("hitl_approved"):
                    logger.info(f"TO-BE HITL activo — pipeline suspendido: {session_id}")
                    return
        else:
            # TO-BE HITL resume: optimization is already done; only BPMN + KPIs remain.
            # Re-invoking the full graph would wastefully re-run extract_asis,
            # analyze_waste and optimize_tobe (no "already done" guards in those nodes),
            # producing a different asis_process/waste_analysis than what the user approved.
            logger.info(f"Reanudando pipeline post-TO-BE HITL: {session_id}")
            current = state.model_dump()
            for node_fn in [node_generate_bpmn, node_calculate_kpis]:
                update = node_fn(AgentState(**current))
                current.update(update)
                _sessions[session_id] = AgentState(**current)

        final_state = _sessions[session_id]
        if final_state.kpi_ok:
            full_report = {
                "asis_process":   final_state.asis_process.model_dump() if final_state.asis_process else None,
                "waste_analysis": final_state.waste_analysis.model_dump() if final_state.waste_analysis else None,
                "tobe_process":   final_state.tobe_process.model_dump() if final_state.tobe_process else None,
                "kpi_report":     final_state.kpi_report.model_dump() if final_state.kpi_report else None,
            }
            score = final_state.waste_analysis.waste_percentage if final_state.waste_analysis else None
            with SessionLocal() as db:
                repo.complete_analysis(db, session_id, full_report, score)
            bpmn_tobe = final_state.bpmn_output.file_path if final_state.bpmn_output else None
            bpmn_asis = final_state.bpmn_asis_output
            if bpmn_tobe or bpmn_asis:
                with SessionLocal() as db:
                    repo.update_bpmn_paths(db, session_id, bpmn_tobe, bpmn_asis)
        logger.info(f"Pipeline reanudado post-HITL: {session_id}")
    except Exception as e:
        logger.error(f"Error al reanudar pipeline {session_id}: {e}")
        if session_id in _sessions:
            _sessions[session_id].errors.append(f"resume_pipeline: {str(e)}")


# ─────────────────────────────────────────────
# HITL — AS-IS REVIEW
# ─────────────────────────────────────────────

@app.get("/sessions/{session_id}/asis-review", tags=["HITL"])
async def get_asis_review(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)
    if not state.hitl_asis_required:
        raise HTTPException(status_code=425, detail="No hay revisión AS-IS pendiente.")
    asis = state.asis_process
    waste = state.waste_analysis
    return {
        "session_id": session_id,
        "asis_process": asis.model_dump() if asis else None,
        "waste_summary": {
            "waste_percentage":    waste.waste_percentage if waste else None,
            "total_waste_time_min": waste.total_waste_time_min if waste else None,
            "top_wastes": [
                {"waste_type": w.waste_type, "activity": w.activity_name}
                for w in (waste.activity_details or [])[:5]
                if w.waste_classification == "desperdicio"
            ] if waste else [],
        },
    }


@app.post("/sessions/{session_id}/asis-review", response_model=SessionResponse, tags=["HITL"])
async def submit_asis_review(
    session_id: str,
    review: HITLReviewRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)

    if not state.hitl_asis_required:
        raise HTTPException(status_code=400, detail="Esta sesión no tiene revisión AS-IS pendiente.")

    state.hitl_asis_approved = review.approved
    state.hitl_asis_feedback = review.feedback or ""
    state.hitl_asis_required = False

    if not review.approved and review.feedback.strip():
        # Re-extract AS-IS with analyst feedback as correction hint
        state.raw_input = state.raw_input + f"\n\n[Corrección del analista: {review.feedback}]"

    _sessions[session_id] = state
    background_tasks.add_task(_resume_pipeline, session_id)

    action = "aprobado" if review.approved else "rechazado — re-extrayendo AS-IS"
    logger.info(f"AS-IS HITL {session_id}: {action}")

    return SessionResponse(
        session_id=session_id,
        message=f"Revisión AS-IS registrada: {action}.",
        current_node=state.current_node,
        status="running",
    )


# ─────────────────────────────────────────────
# RAG — ADMINISTRACIÓN
# ─────────────────────────────────────────────

@app.post("/rag/seed", tags=["RAG"])
async def seed_knowledge_base(current_user: User = Depends(require_admin)):
    try:
        from rag.seed_knowledge import seed
        seed()
        from rag.vector_store import get_collection_stats
        stats = get_collection_stats()
        return {"message": "Knowledge base inicializada correctamente.", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al inicializar knowledge base: {str(e)}")


@app.get("/rag/stats", tags=["RAG"])
async def rag_stats():
    try:
        from rag.vector_store import get_collection_stats
        return get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ─────────────────────────────────────────────
# HISTORIAL — ANÁLISIS PERSISTIDOS
# ─────────────────────────────────────────────

@app.get("/analyses", tags=["Historial"])
async def list_analyses(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
):
    from sqlalchemy import func as _func

    with SessionLocal() as db:
        # Analyses owned by user
        owned = repo.list_analyses(db, limit=limit, offset=offset, user_id=current_user.id)

        # Collaborator counts per owned analysis (single query)
        owned_ids_list = [r.id for r in owned]
        collab_counts: dict[str, int] = {}
        if owned_ids_list:
            rows = (
                db.query(ProcessCollaborator.analysis_id, _func.count(ProcessCollaborator.id))
                .filter(ProcessCollaborator.analysis_id.in_(owned_ids_list))
                .group_by(ProcessCollaborator.analysis_id)
                .all()
            )
            collab_counts = {aid: cnt for aid, cnt in rows}

        # Analyses where user is a collaborator (active or completed)
        collaborated_query = (
            db.query(db_models.Analysis, ProcessCollaborator)
            .join(ProcessCollaborator, ProcessCollaborator.analysis_id == db_models.Analysis.id)
            .filter(
                ProcessCollaborator.user_id == current_user.id,
                ProcessCollaborator.status.in_(["active", "completed"]),
            )
            .all()
        )

        owned_ids = set(owned_ids_list)
        collab_map: dict = {}
        for analysis, collab in collaborated_query:
            if analysis.id not in owned_ids:
                # Resolve owner name for the colaborador card
                owner = db.query(User).filter(User.id == analysis.user_id).first()
                owner_name = owner.full_name if owner else None
                collab_map[analysis.id] = (analysis, collab.status, owner_name)

    def _fmt(r, is_owner: bool, collaborator_status=None, owner_name=None):
        return {
            "id":                       r.id,
            "process_name":             r.process_name,
            "department":               r.department,
            "phase":                    r.phase,
            "status":                   r.status,
            "score":                    r.score,
            "cycle_time_reduction_pct": r.cycle_time_reduction_pct,
            "automation_coverage_pct":  r.automation_coverage_pct,
            "created_at":               r.created_at.isoformat() if r.created_at else None,
            "completed_at":             r.completed_at.isoformat() if r.completed_at else None,
            "has_errors":               r.has_errors,
            "is_owner":                 is_owner,
            "collaborator_status":      collaborator_status,
            "collaborator_count":       collab_counts.get(r.id, 0) if is_owner else None,
            "owner_name":               owner_name,
        }

    analyses = [_fmt(r, is_owner=True) for r in owned]
    analyses += [
        _fmt(r, is_owner=False, collaborator_status=status, owner_name=oname)
        for r, status, oname in collab_map.values()
    ]

    return {"total": len(analyses), "analyses": analyses}


@app.get("/analyses/{session_id}", tags=["Historial"])
async def get_analysis(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    with SessionLocal() as db:
        record = repo.get_analysis(db, session_id, user_id=current_user.id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Análisis '{session_id}' no encontrado.")
    return {
        "id":           record.id,
        "process_name": record.process_name,
        "status":       record.status,
        "created_at":   record.created_at.isoformat() if record.created_at else None,
        "result":       json.loads(record.result_json) if record.result_json else None,
    }


# ─────────────────────────────────────────────
# SESIONES — LIMPIEZA
# ─────────────────────────────────────────────

@app.delete("/sessions/{session_id}", tags=["Sesiones"])
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)
    del _sessions[session_id]
    return {"message": f"Sesión '{session_id}' eliminada."}


@app.get("/sessions", tags=["Sesiones"])
async def list_sessions(current_user: User = Depends(get_current_user)):
    user_sessions = {
        sid: s for sid, s in _sessions.items()
        if not s.user_id or s.user_id == current_user.id
    }
    return {
        "total": len(user_sessions),
        "sessions": [
            {
                "session_id":   sid,
                "current_node": s.current_node,
                "kpi_ok":       s.kpi_ok,
                "errors":       len(s.errors),
            }
            for sid, s in user_sessions.items()
        ],
    }


# ─────────────────────────────────────────────
# SOURCES — per-session document upload
# ─────────────────────────────────────────────

COLAB_DOC_EXTRACTION_PROMPT = """\
Eres un analista de procesos. El siguiente documento fue subido por un colaborador
del proceso "{process_name}". Extrae únicamente la información relevante para
levantar el proceso.

Devuelve un breve resumen en español con esta estructura:
- 🧩 Actividades mencionadas: (lista con bullets)
- ⏱️ Tiempos o duraciones: (si se menciona, si no escribe "no especificado")
- 🛠️ Herramientas / sistemas: (lista)
- 👥 Personas o roles involucrados: (lista)
- ⚠️ Problemas o cuellos de botella: (lista, si se mencionan)

Limita el resumen a 200 palabras. No inventes información que no esté en el texto.

DOCUMENTO:
{document_text}
"""


@app.post("/sessions/{session_id}/sources", tags=["Sources"])
async def upload_source(
    session_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a source document. Behavior depends on role:
    - Consultor / process owner: appends to AgentState.rag_context (legacy behavior).
    - Colaborador: stores under the collab session, runs LLM extraction, posts the
      summary into the collaborator's chat so the agent can react to it.
    """
    is_colaborador = (current_user.business_role or "").lower() == "colaborador"
    filename = file.filename or "upload"

    # ── Colaborador branch ────────────────────────────────────────────────
    if is_colaborador:
        # session_id may be analysis_id or {analysis_id}_{user_id}_collab — normalize
        analysis_id = session_id.split("_")[0] if "_" in session_id else session_id

        with SessionLocal() as db:
            collab = db.query(ProcessCollaborator).filter(
                ProcessCollaborator.analysis_id == analysis_id,
                ProcessCollaborator.user_id == current_user.id,
            ).first()
            if not collab:
                raise HTTPException(status_code=403, detail="You are not a collaborator on this process.")
            analysis = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
            process_name = analysis.process_name if analysis else "este proceso"

        collab_sid = f"{analysis_id}_{current_user.id}_collab"
        sources_dir = Path(f"storage/outputs/sources/{analysis_id}/colab/{current_user.id}")
        sources_dir.mkdir(parents=True, exist_ok=True)
        file_path = sources_dir / filename
        file_path.write_bytes(await file.read())

        try:
            extracted_text = load_document(str(file_path))
        except (ValueError, FileNotFoundError) as e:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Cannot extract text from file: {e}")

        # LLM extraction — focused on process-relevant fields
        summary = ""
        try:
            from llm.factory import get_llm
            from langchain_core.messages import HumanMessage
            llm = get_llm(temperature=0.2)
            prompt = COLAB_DOC_EXTRACTION_PROMPT.format(
                process_name=process_name,
                document_text=extracted_text[:6000],
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            summary = (response.content or "").strip()
        except Exception as exc:
            logger.warning(f"Colab doc extraction LLM failed (analysis={analysis_id}): {exc}")
            summary = (
                "He recibido el documento, pero no pude procesarlo automáticamente. "
                "¿Podrías contarme qué actividades de tu proceso aparecen ahí?"
            )

        # Post assistant message into the collab chat
        chat_msg = (
            f"He procesado tu documento **{filename}**.\n\n"
            f"{summary}\n\n"
            f"¿Confirmas que estas actividades forman parte de tu participación en el proceso?"
        )
        with SessionLocal() as db:
            repo.save_chat_message(db, collab_sid, "assistant", chat_msg)
            # Mark the contribution as RAG-indexed (tenant-only). Vector indexing of
            # the snippet is future work — the flag tracks intent for now.
            collab_db = db.query(ProcessCollaborator).filter(
                ProcessCollaborator.analysis_id == analysis_id,
                ProcessCollaborator.user_id == current_user.id,
            ).first()
            if collab_db:
                collab_db.rag_indexed = True
                db.commit()

        meta = {
            "filename":    filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "chars":       len(extracted_text),
        }
        _session_sources.setdefault(collab_sid, []).append(meta)
        logger.info(f"Colab source uploaded: analysis={analysis_id} user={current_user.id} file={filename}")
        return {
            "filename": filename,
            "extracted_chars": len(extracted_text),
            "status": "processed",
            "chat_session_id": collab_sid,
        }

    # ── Consultor / owner branch (legacy) ─────────────────────────────────
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)

    sources_dir = Path(f"storage/outputs/sources/{session_id}")
    sources_dir.mkdir(parents=True, exist_ok=True)
    file_path = sources_dir / filename
    file_path.write_bytes(await file.read())

    try:
        extracted_text = load_document(str(file_path))
    except (ValueError, FileNotFoundError) as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Cannot extract text from file: {e}")

    snippet = extracted_text[:2000]
    state.rag_context.append(f"\n\nAdditional source ({filename}): {snippet}")
    state.additional_sources.append(snippet)
    _sessions[session_id] = state

    meta = {
        "filename":    filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "chars":       len(extracted_text),
    }
    _session_sources.setdefault(session_id, []).append(meta)

    logger.info(f"Source uploaded for session {session_id}: {filename} ({len(extracted_text)} chars)")
    return {"filename": filename, "extracted_chars": len(extracted_text), "status": "added"}


@app.get("/sessions/{session_id}/sources", tags=["Sources"])
async def list_sources(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Return list of uploaded source files for a session."""
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)
    return {"sources": _session_sources.get(session_id, [])}


# ─────────────────────────────────────────────
# REFINEMENT — modify TO-BE without re-running full pipeline
# ─────────────────────────────────────────────

@app.post("/sessions/{session_id}/refine", tags=["Refinement"])
async def refine_session(
    session_id: str,
    body: RefineRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Modify the TO-BE process with a natural-language instruction,
    then regenerate the BPMN and recalculate KPIs — without restarting the pipeline.
    Only works when kpi_ok is True.
    """
    from agent.refiner import refine_tobe_process
    from agent.bpmn_generator import node_generate_bpmn
    from agent.kpi_calculator import node_calculate_kpis

    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)

    if not state.kpi_ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pipeline must be complete (kpi_ok=True) before refinement.",
        )
    if not state.tobe_process:
        raise HTTPException(status_code=400, detail="No TO-BE process found in session.")

    try:
        new_tobe = refine_tobe_process(state.tobe_process, body.instruction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement LLM error: {e}")

    state.tobe_process  = new_tobe
    state.hitl_approved = True  # bypass HITL gate in node_generate_bpmn

    bpmn_update = node_generate_bpmn(state)
    for k, v in bpmn_update.items():
        if hasattr(state, k):
            setattr(state, k, v)

    kpi_update = node_calculate_kpis(state)
    for k, v in kpi_update.items():
        if hasattr(state, k):
            setattr(state, k, v)

    _sessions[session_id] = state

    logger.info(f"Session {session_id} refined: bpmn_ok={state.bpmn_ok}, kpi_ok={state.kpi_ok}")
    return {
        "tobe_process": state.tobe_process.model_dump(),
        "bpmn_updated": state.bpmn_ok,
        "kpis_updated": state.kpi_report.model_dump() if state.kpi_report else None,
    }


# ─────────────────────────────────────────────
# COLLABORATION — Process Collaborators
# ─────────────────────────────────────────────

class InviteCollaboratorRequest(BaseModel):
    email: str = Field(..., description="Email of the person to invite")
    message: str = Field(default="", description="Optional personal message")


@app.post("/analyses/{analysis_id}/invite", tags=["Collaboration"])
async def invite_collaborator(
    analysis_id: str,
    body: InviteCollaboratorRequest,
    current_user: User = Depends(get_current_user),
):
    import json as _json
    from datetime import timedelta

    with SessionLocal() as db:
        analysis = db.query(db_models.Analysis).filter(
            db_models.Analysis.id == analysis_id,
            db_models.Analysis.tenant_id == current_user.tenant_id,
        ).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only the process owner can invite collaborators.")

        # Check if invitee already exists in same tenant
        invitee = db.query(User).filter(
            User.email == body.email,
            User.tenant_id == current_user.tenant_id,
        ).first()

        if invitee:
            # Create collaborator record (pending)
            collab = ProcessCollaborator(
                analysis_id=analysis_id,
                user_id=invitee.id,
                invited_by=current_user.id,
                status="pending",
            )
            db.add(collab)

            # Notify the invitee in-app
            notif = Notification(
                user_id=invitee.id,
                type="process_invitation",
                title=f"Invitación a colaborar en {analysis.process_name}",
                message=f"{current_user.full_name} te invitó a colaborar",
                data=_json.dumps({
                    "analysis_id": analysis_id,
                    "analysis_name": analysis.process_name,
                    "invited_by_name": current_user.full_name,
                }),
            )
            db.add(notif)
            db.commit()

            logger.info(f"Collaboration invited (existing user): analysis={analysis_id} invitee={invitee.email}")
            return {"status": "invited", "user_found": True}

        else:
            # Create external invitation with a unique token
            token = str(uuid.uuid4())
            invitation = Invitation(
                email=body.email,
                analysis_id=analysis_id,
                invited_by=current_user.id,
                role="colaborador",
                token=token,
                status="pending",
                expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            )
            db.add(invitation)
            db.commit()

            invite_url = f"/accept-invite/{token}"
            logger.info(f"Invitation created (external): analysis={analysis_id} email={body.email}")
            return {
                "status": "invited",
                "user_found": False,
                "invite_token": token,
                "invite_url": invite_url,
            }


@app.get("/analyses/{analysis_id}/collaborators", tags=["Collaboration"])
async def list_collaborators(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
):
    with SessionLocal() as db:
        analysis = db.query(db_models.Analysis).filter(
            db_models.Analysis.id == analysis_id,
            db_models.Analysis.tenant_id == current_user.tenant_id,
        ).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only the process owner can view collaborators.")

        collabs = db.query(ProcessCollaborator).filter(
            ProcessCollaborator.analysis_id == analysis_id,
        ).all()

        result = []
        for c in collabs:
            user = db.query(User).filter(User.id == c.user_id).first()
            result.append({
                "user_id": c.user_id,
                "full_name": user.full_name if user else "Unknown",
                "email": user.email if user else "Unknown",
                "status": c.status,
                "completed_at": c.completed_at.isoformat() if c.completed_at else None,
            })

    return {"collaborators": result}


def _build_contribution_summary(chat_history: list[dict]) -> str:
    """Concatenate a collaborator's chat into a compact textual contribution."""
    if not chat_history:
        return ""
    lines = []
    for m in chat_history:
        role = "Colaborador" if m.get("role") == "user" else "Asistente"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


@app.post("/analyses/{analysis_id}/collaborators/{user_id}/complete", tags=["Collaboration"])
async def complete_collaboration(
    analysis_id: str,
    user_id: str,
    current_user: User = Depends(get_current_user),
):
    import json as _json

    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="You can only complete your own collaboration.")

    with SessionLocal() as db:
        collab = db.query(ProcessCollaborator).filter(
            ProcessCollaborator.analysis_id == analysis_id,
            ProcessCollaborator.user_id == user_id,
        ).first()
        if not collab:
            raise HTTPException(status_code=404, detail="Collaboration record not found.")

        analysis = db.query(db_models.Analysis).filter(
            db_models.Analysis.id == analysis_id,
        ).first()

        collab.status = "completed"
        collab.completed_at = datetime.now(timezone.utc)

        # Capture chat session and build contribution summary from messages
        session_id = collab.session_id or f"{analysis_id}_{user_id}_collab"
        collab.session_id = session_id
        chat_history = repo.get_chat_messages(db, session_id)
        collab.contribution_summary = _build_contribution_summary(chat_history)

        # Notify the analysis owner
        if analysis and analysis.user_id:
            notif = Notification(
                user_id=analysis.user_id,
                type="collaborator_completed",
                title=f"{current_user.full_name} completó su colaboración",
                message=f"Ya puedes ver su aporte en {analysis.process_name}",
                data=_json.dumps({
                    "analysis_id": analysis_id,
                    "collaborator_id": user_id,
                    "collaborator_name": current_user.full_name,
                }),
            )
            db.add(notif)

        db.commit()
        logger.info(f"Collaboration completed: analysis={analysis_id} user={user_id}")

    return {"status": "completed"}


@app.get(
    "/analyses/{analysis_id}/collaborators/{user_id}/contribution",
    tags=["Collaboration"],
)
async def get_collaborator_contribution(
    analysis_id: str,
    user_id: str,
    current_user: User = Depends(get_current_user),
):
    """Owner-only view of a collaborator's chat history and contribution summary."""
    with SessionLocal() as db:
        analysis = db.query(db_models.Analysis).filter(
            db_models.Analysis.id == analysis_id,
            db_models.Analysis.tenant_id == current_user.tenant_id,
        ).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only the process owner can view contributions.")

        collab = db.query(ProcessCollaborator).filter(
            ProcessCollaborator.analysis_id == analysis_id,
            ProcessCollaborator.user_id == user_id,
        ).first()
        if not collab:
            raise HTTPException(status_code=404, detail="Collaboration record not found.")

        collaborator = db.query(User).filter(User.id == user_id).first()
        session_id = collab.session_id or f"{analysis_id}_{user_id}_collab"

        rows = (
            db.query(db_models.ChatMessage)
            .filter(db_models.ChatMessage.session_id == session_id)
            .order_by(db_models.ChatMessage.created_at)
            .all()
        )
        chat_history = [
            {
                "role": r.role,
                "content": r.content,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    return {
        "collaborator": {
            "name":  collaborator.full_name if collaborator else "Unknown",
            "email": collaborator.email if collaborator else "Unknown",
            "status": collab.status,
            "completed_at": collab.completed_at.isoformat() if collab.completed_at else None,
        },
        "chat_history": chat_history,
        "contribution_summary": collab.contribution_summary,
    }


UNIFY_SYSTEM_PROMPT = """\
You are a process analysis expert.
Multiple team members have described their parts of the process "{process_name}".

Here are their contributions:
{contributions}

Your task:
1. Identify all unique activities mentioned.
2. Remove duplicates and overlaps.
3. Order activities in logical sequence.
4. Estimate total process duration.
5. Return a unified AS-IS process in this JSON format:
{{
  "name": "process name",
  "description": "unified description",
  "department": "department",
  "objective": "objective",
  "activities": [
    {{
      "name": "activity name",
      "description": "description",
      "responsible": "role/person",
      "estimated_duration_min": 15,
      "activity_type": "operativa|analitica|cognitiva",
      "dependencies": []
    }}
  ]
}}

Return ONLY valid JSON, no markdown, no explanations.
"""


@app.post("/analyses/{analysis_id}/unify", tags=["Collaboration"])
async def unify_contributions(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
):
    """Synthesize a unified AS-IS process from all completed collaborator chats."""
    from llm.factory import get_llm
    import json as _json
    import re

    with SessionLocal() as db:
        analysis = db.query(db_models.Analysis).filter(
            db_models.Analysis.id == analysis_id,
            db_models.Analysis.tenant_id == current_user.tenant_id,
        ).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only the process owner can unify contributions.")

        completed = db.query(ProcessCollaborator).filter(
            ProcessCollaborator.analysis_id == analysis_id,
            ProcessCollaborator.status == "completed",
        ).all()
        if not completed:
            raise HTTPException(
                status_code=400,
                detail="At least one collaborator must complete their contribution before unifying.",
            )

        # Build per-collaborator contribution blocks
        blocks = []
        for idx, c in enumerate(completed, start=1):
            user = db.query(User).filter(User.id == c.user_id).first()
            name = user.full_name if user else f"Collaborator {idx}"
            session_id = c.session_id or f"{analysis_id}_{c.user_id}_collab"
            history = repo.get_chat_messages(db, session_id)
            summary = c.contribution_summary or _build_contribution_summary(history)
            if not summary:
                continue
            blocks.append(f"COLLABORATOR {idx} ({name}):\n{summary}")

        if not blocks:
            raise HTTPException(
                status_code=400,
                detail="No collaborator chat content available to unify.",
            )

        prompt = UNIFY_SYSTEM_PROMPT.format(
            process_name=analysis.process_name or "Unknown",
            contributions="\n\n".join(blocks),
        )

    # LLM call outside the DB session
    try:
        from langchain_core.messages import HumanMessage
        llm = get_llm(temperature=0.2)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content or "").strip()
    except Exception as e:
        logger.error(f"LLM error during unify (analysis={analysis_id}): {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM failed to synthesize the unified process: {e}",
        )

    # Strip code fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        unified = _json.loads(raw)
    except _json.JSONDecodeError as e:
        logger.error(f"Unify produced invalid JSON (analysis={analysis_id}): {e}\nRaw: {raw[:500]}")
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON. Try again or adjust the collaborator chats.",
        )

    # Persist: store unified AS-IS into result_json, and update raw_input for future reruns
    with SessionLocal() as db:
        record = db.query(db_models.Analysis).filter(
            db_models.Analysis.id == analysis_id,
        ).first()
        if record:
            existing = _json.loads(record.result_json) if record.result_json else {}
            existing["asis_process"] = unified
            existing["unified_from_collaborators"] = True
            record.result_json = _json.dumps(existing, default=str)
            # Rebuild a textual raw_input so the pipeline can re-extract if needed
            record.raw_input = "\n\n".join(blocks)
            db.commit()
            logger.info(f"Unified AS-IS generated for analysis={analysis_id} from {len(blocks)} collaborator(s)")

    return {"unified_asis": unified, "message": "Unified successfully"}


# ─────────────────────────────────────────────
# SPRINT 8 — PIPELINE STATE MACHINE
# (collecting → unifying → asis_hitl → optimizing → tobe_hitl → completed)
# ─────────────────────────────────────────────

SPRINT8_UNIFY_PROMPT = """\
Eres un analista experto en levantamiento y consolidación de procesos empresariales.
Varios miembros del área "{department}" han descrito sus partes del proceso "{process_name}".

{consultant_context}

CONTRIBUCIONES DE LOS COLABORADORES:
{contributions}

{revision_block}

TU TAREA:
1. Identifica TODAS las actividades únicas mencionadas por los colaboradores.
2. Elimina duplicados (cuando dos personas describen el mismo paso, consolida en uno).
3. Ordena las actividades en una secuencia de negocio lógica.
4. Estima la duración realista de cada actividad según lo descrito.
5. Identifica el responsable (rol o cargo) de cada actividad.
6. Captura los puntos de dolor, sistemas usados, y dependencias.

Devuelve un JSON con esta estructura exacta:
{{
  "name": "{process_name}",
  "department": "{department}",
  "objective": "objetivo principal del proceso",
  "scope": "alcance: punto de inicio y fin",
  "total_duration_min": 0,
  "activities": [
    {{
      "id": "ACT-001",
      "name": "nombre de la actividad",
      "description": "qué sucede en este paso",
      "responsible": "rol o persona responsable",
      "estimated_duration_min": 30,
      "activity_type": "operativa",
      "tools_used": ["Excel", "SAP"],
      "pain_points": ["proceso manual", "propenso a errores"],
      "depends_on": []
    }}
  ],
  "identified_bottlenecks": ["cuellos de botella identificados"],
  "consolidation_notes": "explicación breve de cómo se unificaron las contribuciones"
}}

activity_type debe ser uno de: operativa | analitica | cognitiva.

Devuelve SOLO JSON válido. Sin markdown, sin explicaciones, sin texto extra.
"""


def _strip_json_fence(raw: str) -> str:
    """Remove ```json fences if the LLM ignored instructions."""
    import re
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _format_asis_for_chat(asis: dict) -> str:
    """Build a Spanish natural-language message describing the unified AS-IS."""
    name = asis.get("name", "—")
    objective = asis.get("objective", "—")
    department = asis.get("department", "—")
    total = asis.get("total_duration_min") or sum(
        (a.get("estimated_duration_min") or 0) for a in asis.get("activities", [])
    )
    acts = asis.get("activities", [])
    bottlenecks = asis.get("identified_bottlenecks", []) or []
    notes = asis.get("consolidation_notes", "")

    act_lines = []
    for i, a in enumerate(acts, start=1):
        dur = a.get("estimated_duration_min", "?")
        act_lines.append(
            f"  {i}. **{a.get('name','—')}** ({dur} min)\n"
            f"     Responsable: {a.get('responsible','—')}\n"
            f"     {a.get('description','')}"
        )

    bn_block = "\n".join(f"• {b}" for b in bottlenecks) if bottlenecks else "_(ninguno identificado)_"

    return (
        f"He analizado todas las contribuciones del equipo y consolidé el AS-IS del proceso "
        f"**{name}**.\n\n"
        f"**Resumen del proceso:**\n"
        f"- 📋 Objetivo: {objective}\n"
        f"- 🏢 Departamento: {department}\n"
        f"- ⏱️ Duración total estimada: {total} minutos\n"
        f"- 📊 Actividades identificadas: {len(acts)}\n\n"
        f"**Actividades del proceso:**\n"
        f"{chr(10).join(act_lines)}\n\n"
        f"**Cuellos de botella identificados:**\n"
        f"{bn_block}\n\n"
        f"**Nota de consolidación:** {notes}\n\n"
        f"---\n"
        f"¿Apruebas este AS-IS o necesitas algún ajuste? "
        f"Usa los botones del panel izquierdo, o describe los cambios en el chat."
    )


def _format_tobe_for_chat(tobe: "TOBEProcess", waste, rag_count: int) -> str:
    """Build a Spanish natural-language message describing the optimized TO-BE."""
    methodologies = ", ".join(tobe.applied_methodologies or [])
    duration = tobe.total_duration_min or 0
    acts = tobe.activities or []

    status_icon = {
        "conservada": "✅",
        "optimizada": "⚡",
        "automatizada": "🤖",
        "eliminada":   "🗑️",
        "combinada":   "🔀",
        "nueva":       "✨",
    }
    waste_label = {
        "espera": "Espera",
        "sobreproceso": "Sobreproceso",
        "defectos": "Defectos",
        "sobreproduccion": "Sobreproducción",
        "transporte": "Transporte",
        "inventario": "Inventario",
        "movimiento": "Movimiento",
        "talento_no_utilizado": "Talento no utilizado",
    }

    act_lines = []
    for i, a in enumerate(acts, start=1):
        status = (a.status.value if hasattr(a.status, "value") else str(a.status)) or "conservada"
        icon = status_icon.get(status, "•")
        tool = f" — herramienta: **{a.automation_tool}**" if a.automation_tool else ""
        dur = a.estimated_duration_min if a.estimated_duration_min is not None else 0
        act_lines.append(
            f"  {i}. {icon} **{a.name}** ({dur} min, {status}){tool}\n"
            f"     {a.improvement_justification}"
        )

    waste_lines = []
    if waste and getattr(waste, "activity_details", None):
        wastes = [d for d in waste.activity_details if d.waste_classification.value == "desperdicio"]
        for w in wastes[:6]:
            wt = w.waste_type.value if w.waste_type else "—"
            waste_lines.append(f"• **{w.activity_name}** — {waste_label.get(wt, wt)}")

    qw_lines = []
    if waste and getattr(waste, "kaizen_quick_wins", None):
        for q in (waste.kaizen_quick_wins or [])[:5]:
            qw_lines.append(f"• {q}")

    return (
        f"Apliqué análisis Lean / Six Sigma / Kaizen sobre el AS-IS aprobado.\n\n"
        f"**Resumen de la optimización TO-BE:**\n"
        f"- 🧠 Metodologías aplicadas: {methodologies or 'Lean, Six Sigma, Kaizen'}\n"
        f"- ⏱️ Duración total optimizada: {duration} min\n"
        f"- 📊 Actividades en el TO-BE: {len(acts)}\n"
        f"- 📚 Patrones RAG consultados: {rag_count}\n\n"
        f"**Actividades del TO-BE:**\n"
        f"{chr(10).join(act_lines) if act_lines else '_(sin actividades)_'}\n\n"
        f"**Desperdicios (Muda) detectados:**\n"
        f"{chr(10).join(waste_lines) if waste_lines else '_(no se detectaron)_'}\n\n"
        f"**Quick wins Kaizen:**\n"
        f"{chr(10).join(qw_lines) if qw_lines else '_(ninguno)_'}\n\n"
        f"---\n"
        f"¿Apruebas el TO-BE para generar el reporte final, o necesitas ajustes? "
        f"Usa los botones del panel izquierdo, o describe los cambios en el chat."
    )


def _collect_contributions(db, analysis_id: str) -> tuple[list[str], list[str]]:
    """Returns (blocks, collaborator_names) for completed collaborators on this analysis."""
    completed = db.query(ProcessCollaborator).filter(
        ProcessCollaborator.analysis_id == analysis_id,
        ProcessCollaborator.status == "completed",
    ).all()
    blocks: list[str] = []
    names:  list[str] = []
    for idx, c in enumerate(completed, start=1):
        user = db.query(User).filter(User.id == c.user_id).first()
        cname = user.full_name if user else f"Collaborator {idx}"
        sid = c.session_id or f"{analysis_id}_{c.user_id}_collab"
        history = repo.get_chat_messages(db, sid)
        summary = c.contribution_summary or _build_contribution_summary(history)
        if not summary:
            continue
        blocks.append(f"COLABORADOR {idx} ({cname}):\n{summary}")
        names.append(cname)
    return blocks, names


def _build_process_from_unified(unified: dict, analysis_id: str, fallback_name: str) -> "Process":
    """Convert the LLM's unified JSON into a Process pydantic model."""
    from models.schemas import Process, Activity, ActivityType
    from datetime import datetime as _dt
    activities: list[Activity] = []
    for i, raw_act in enumerate(unified.get("activities", []), start=1):
        atype_raw = (raw_act.get("activity_type") or raw_act.get("type") or "operativa").lower()
        try:
            atype = ActivityType(atype_raw)
        except ValueError:
            atype = ActivityType.OPERATIVE
        activities.append(Activity(
            id=raw_act.get("id") or f"ACT-{i:03d}",
            name=raw_act.get("name") or f"Actividad {i}",
            description=raw_act.get("description") or "",
            responsible=raw_act.get("responsible") or "—",
            type=atype,
            estimated_duration_min=raw_act.get("estimated_duration_min"),
            depends_on=raw_act.get("depends_on") or raw_act.get("dependencies") or [],
            systems_used=raw_act.get("tools_used") or raw_act.get("systems_used") or [],
        ))
    total = unified.get("total_duration_min") or sum(
        (a.estimated_duration_min or 0) for a in activities
    )
    return Process(
        id=f"ASIS-{analysis_id[:8]}",
        name=unified.get("name") or fallback_name or "Proceso",
        description=unified.get("objective") or unified.get("description") or "",
        owner=unified.get("department") or "",
        scope=unified.get("scope") or "Desde inicio hasta cierre del proceso",
        participants=[],
        systems=[],
        activities=activities,
        total_duration_min=float(total or 0),
        raw_input=unified.get("consolidation_notes") or "",
        extracted_at=_dt.utcnow(),
    )


async def _run_unification(analysis_id: str, owner_user_id: str, feedback: str = "") -> None:
    """Background task — calls the LLM to consolidate collaborator chats into a unified AS-IS."""
    from llm.factory import get_llm
    from langchain_core.messages import HumanMessage
    import json as _json

    try:
        with SessionLocal() as db:
            record = db.query(db_models.Analysis).filter(
                db_models.Analysis.id == analysis_id,
            ).first()
            if not record:
                logger.error(f"Unification aborted — analysis {analysis_id} not found")
                return
            process_name = record.process_name or "Proceso sin nombre"
            department = record.department or "—"
            consultant_desc = record.raw_input or ""
            blocks, _names = _collect_contributions(db, analysis_id)

        if not blocks:
            logger.warning(f"Unification: no contributions for {analysis_id}")
            with SessionLocal() as db:
                rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
                if rec:
                    rec.phase = "collecting"
                    db.commit()
            return

        consultant_block = (
            f"DESCRIPCIÓN INICIAL DEL CONSULTOR:\n{consultant_desc}\n"
            if consultant_desc.strip() else ""
        )
        revision_block = (
            f"NOTA DE REVISIÓN DEL CONSULTOR — la versión anterior debe corregirse así:\n{feedback}\n"
            if feedback.strip() else ""
        )
        prompt = SPRINT8_UNIFY_PROMPT.format(
            process_name=process_name,
            department=department,
            consultant_context=consultant_block,
            contributions="\n\n".join(blocks),
            revision_block=revision_block,
        )

        llm = get_llm(temperature=0.2)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = _strip_json_fence(response.content or "")
        try:
            unified = _json.loads(raw)
        except _json.JSONDecodeError:
            logger.error(f"Unify {analysis_id}: invalid JSON\nRaw: {raw[:500]}")
            with SessionLocal() as db:
                rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
                if rec:
                    rec.phase = "collecting"
                    db.commit()
                repo.save_chat_message(db, analysis_id, "assistant",
                    "❌ No pude consolidar las contribuciones — la respuesta del LLM no era válida. "
                    "Vuelve a intentar la unificación.")
            return

        # Persist AS-IS in DB and AgentState in memory
        asis_process = _build_process_from_unified(unified, analysis_id, process_name)
        with SessionLocal() as db:
            rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
            if rec:
                existing = _json.loads(rec.result_json) if rec.result_json else {}
                existing["asis_process"] = asis_process.model_dump(mode="json")
                existing["unified_from_collaborators"] = True
                rec.result_json = _json.dumps(existing, default=str)
                rec.raw_input = "\n\n".join(blocks)
                rec.phase = "asis_hitl"
                db.commit()
            repo.save_chat_message(db, analysis_id, "assistant", _format_asis_for_chat(unified))

        state = _sessions.get(analysis_id) or AgentState(
            raw_input="\n\n".join(blocks),
            user_id=owner_user_id,
            current_node="hitl_review_asis",
        )
        state.asis_process = asis_process
        state.extraction_ok = True
        state.hitl_asis_required = True
        state.hitl_asis_approved = False
        state.current_node = "hitl_review_asis"
        _sessions[analysis_id] = state

        logger.info(f"Unification done: {analysis_id} ({len(unified.get('activities',[]))} activities)")
    except Exception as e:
        logger.exception(f"_run_unification failed for {analysis_id}: {e}")


async def _run_optimization(analysis_id: str, feedback: str = "") -> None:
    """Background task — runs RAG + waste analysis + TO-BE generation on the approved AS-IS."""
    from agent.analyzer import node_analyze_waste
    from rag.retriever import node_retrieve_rag
    from agent.optimizer import node_optimize_tobe

    try:
        state = _sessions.get(analysis_id)
        if state is None or state.asis_process is None:
            # Recover AS-IS from DB
            with SessionLocal() as db:
                rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
            if not rec or not rec.result_json:
                logger.error(f"Optimization aborted — no AS-IS for {analysis_id}")
                return
            existing = json.loads(rec.result_json)
            from models.schemas import Process
            asis = Process(**existing["asis_process"])
            state = AgentState(
                raw_input=rec.raw_input or "",
                asis_process=asis,
                extraction_ok=True,
                hitl_asis_approved=True,
                user_id=rec.user_id,
                tenant_id=rec.tenant_id,
                current_node="analyze_waste",
            )
            _sessions[analysis_id] = state

        if feedback.strip():
            state.hitl_feedback = feedback
            state.hitl_retries = (state.hitl_retries or 0) + 1

        # 1) Waste analysis
        upd = node_analyze_waste(state)
        for k, v in upd.items():
            if hasattr(state, k):
                setattr(state, k, v)

        # 2) RAG retrieval
        upd = node_retrieve_rag(state)
        for k, v in upd.items():
            if hasattr(state, k):
                setattr(state, k, v)

        # 3) TO-BE generation
        upd = node_optimize_tobe(state)
        for k, v in upd.items():
            if hasattr(state, k):
                setattr(state, k, v)

        if not state.optimization_ok or state.tobe_process is None:
            logger.error(f"Optimization failed for {analysis_id}: {state.errors[-3:]}")
            with SessionLocal() as db:
                rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
                if rec:
                    rec.phase = "asis_hitl"
                    db.commit()
                repo.save_chat_message(db, analysis_id, "assistant",
                    "❌ La optimización falló. Puedes solicitar una nueva revisión del AS-IS o reintentar la aprobación.")
            _sessions[analysis_id] = state
            return

        chat_msg = _format_tobe_for_chat(
            state.tobe_process,
            state.waste_analysis,
            rag_count=len(state.rag_context or []),
        )

        # Persist TO-BE
        with SessionLocal() as db:
            rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
            if rec:
                existing = json.loads(rec.result_json) if rec.result_json else {}
                existing["waste_analysis"] = state.waste_analysis.model_dump(mode="json") if state.waste_analysis else None
                existing["tobe_process"]   = state.tobe_process.model_dump(mode="json")
                rec.result_json = json.dumps(existing, default=str)
                rec.phase = "tobe_hitl"
                db.commit()
            repo.save_chat_message(db, analysis_id, "assistant", chat_msg)

        state.hitl_required = True
        state.hitl_approved = False
        state.current_node = "hitl_review"
        _sessions[analysis_id] = state
        logger.info(f"Optimization done: {analysis_id} — TO-BE has {len(state.tobe_process.activities)} activities")
    except Exception as e:
        logger.exception(f"_run_optimization failed for {analysis_id}: {e}")


async def _generate_final_report(analysis_id: str) -> None:
    """Background task — generates BPMN (AS-IS + TO-BE) and final KPIs after TO-BE approval."""
    from agent.bpmn_generator import node_generate_asis_bpmn, node_generate_bpmn
    from agent.kpi_calculator import node_calculate_kpis

    try:
        state = _sessions.get(analysis_id)
        if state is None or state.tobe_process is None:
            logger.error(f"Final report aborted — TO-BE not in memory for {analysis_id}")
            return

        state.hitl_approved = True
        if state.tobe_process:
            state.tobe_process.human_approved = True

        for node_fn in (node_generate_asis_bpmn, node_generate_bpmn, node_calculate_kpis):
            upd = node_fn(state)
            for k, v in upd.items():
                if hasattr(state, k):
                    setattr(state, k, v)

        _sessions[analysis_id] = state

        # Sprint 8 — Muda/Mura/Muri classification + tool recommendations
        enrichment = None
        try:
            from agent.waste_enrichment import build_enrichment_block
            if state.asis_process and state.tobe_process and state.waste_analysis:
                enrichment = build_enrichment_block(
                    state.asis_process, state.tobe_process, state.waste_analysis,
                )
        except Exception as exc:
            logger.warning(f"waste_enrichment failed for {analysis_id}: {exc}")

        full_report = {
            "asis_process":   state.asis_process.model_dump(mode="json") if state.asis_process else None,
            "waste_analysis": state.waste_analysis.model_dump(mode="json") if state.waste_analysis else None,
            "tobe_process":   state.tobe_process.model_dump(mode="json") if state.tobe_process else None,
            "kpi_report":     state.kpi_report.model_dump(mode="json") if state.kpi_report else None,
            "unified_from_collaborators": True,
            "enrichment":     enrichment,
        }
        score = state.waste_analysis.waste_percentage if state.waste_analysis else None

        with SessionLocal() as db:
            repo.complete_analysis(db, analysis_id, full_report, score)
            bpmn_tobe = state.bpmn_output.file_path if state.bpmn_output else None
            bpmn_asis = state.bpmn_asis_output
            if bpmn_tobe or bpmn_asis:
                repo.update_bpmn_paths(db, analysis_id, bpmn_tobe, bpmn_asis)
            rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
            if rec:
                rec.phase = "completed"
                db.commit()
            repo.save_chat_message(db, analysis_id, "assistant",
                "🎉 ¡Reporte final generado! Ya puedes ver el BPMN, las métricas y las recomendaciones "
                "en los paneles de la derecha.")
        logger.info(f"Final report completed: {analysis_id}")
    except Exception as e:
        logger.exception(f"_generate_final_report failed for {analysis_id}: {e}")


def _assert_owner_and_phase(analysis_id: str, current_user: User, required_phases: tuple[str, ...]) -> db_models.Analysis:
    """Returns the analysis if user owns it and phase is one of required_phases. Otherwise raises."""
    with SessionLocal() as db:
        rec = db.query(db_models.Analysis).filter(
            db_models.Analysis.id == analysis_id,
            db_models.Analysis.tenant_id == current_user.tenant_id,
        ).first()
        if not rec:
            raise HTTPException(status_code=404, detail="Process not found.")
        if rec.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only the process owner can perform this action.")
        if rec.phase not in required_phases:
            raise HTTPException(
                status_code=409,
                detail=f"Process is in phase '{rec.phase}', expected one of {required_phases}.",
            )
        return rec


@app.post("/processes/{analysis_id}/start-unification", tags=["Processes"])
async def start_unification(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Consolidate collaborator chats into a unified AS-IS. Posts result to consultant chat."""
    rec = _assert_owner_and_phase(analysis_id, current_user, ("collecting", "asis_hitl"))

    with SessionLocal() as db:
        completed = db.query(ProcessCollaborator).filter(
            ProcessCollaborator.analysis_id == analysis_id,
            ProcessCollaborator.status == "completed",
        ).count()
        if completed == 0:
            raise HTTPException(
                status_code=400,
                detail="Al menos un colaborador debe completar su contribución antes de unificar.",
            )
        rec.phase = "unifying"
        db.commit()

    background_tasks.add_task(_run_unification, analysis_id, current_user.id, "")
    return {"status": "unifying", "message": "Unificación iniciada"}


@app.post("/processes/{analysis_id}/approve-asis", tags=["Processes"])
async def approve_asis_sprint8(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Approve the unified AS-IS and kick off the optimization pipeline."""
    _assert_owner_and_phase(analysis_id, current_user, ("asis_hitl",))
    with SessionLocal() as db:
        rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
        rec.phase = "optimizing"
        db.commit()
        repo.save_chat_message(db, analysis_id, "system",
            "✅ AS-IS aprobado. Iniciando análisis Lean / Six Sigma / Kaizen…")

    background_tasks.add_task(_run_optimization, analysis_id, "")
    return {"status": "optimizing", "message": "Optimización iniciada"}


@app.post("/processes/{analysis_id}/approve-tobe", tags=["Processes"])
async def approve_tobe_sprint8(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Approve the TO-BE and generate BPMN + KPIs for the final report."""
    _assert_owner_and_phase(analysis_id, current_user, ("tobe_hitl",))
    with SessionLocal() as db:
        rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
        # Stay in tobe_hitl until generation finishes; the background task flips to "completed"
        repo.save_chat_message(db, analysis_id, "system",
            "✅ TO-BE aprobado. Generando BPMN y reporte final…")

    background_tasks.add_task(_generate_final_report, analysis_id)
    return {"status": "generating_report", "message": "Reporte en generación"}


@app.post("/processes/{analysis_id}/request-revision", tags=["Processes"])
async def request_revision_sprint8(
    analysis_id: str,
    body: RevisionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Reject the current AS-IS or TO-BE. The agent regenerates with the consultant feedback."""
    if body.phase not in ("asis_hitl", "tobe_hitl"):
        raise HTTPException(status_code=422, detail="phase must be 'asis_hitl' or 'tobe_hitl'.")
    _assert_owner_and_phase(analysis_id, current_user, (body.phase,))

    with SessionLocal() as db:
        rec = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
        if body.phase == "asis_hitl":
            rec.phase = "unifying"
        else:
            rec.phase = "optimizing"
        db.commit()
        repo.save_chat_message(db, analysis_id, "user", body.feedback)
        repo.save_chat_message(db, analysis_id, "system",
            "🔄 Regenerando con tus indicaciones…")

    if body.phase == "asis_hitl":
        background_tasks.add_task(_run_unification, analysis_id, current_user.id, body.feedback)
    else:
        background_tasks.add_task(_run_optimization, analysis_id, body.feedback)

    return {"status": "regenerating", "phase": body.phase}


# ─────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────

@app.get("/notifications", tags=["Notifications"])
async def list_notifications(
    unread_only: bool = False,
    current_user: User = Depends(get_current_user),
):
    with SessionLocal() as db:
        q = db.query(Notification).filter(Notification.user_id == current_user.id)
        if unread_only:
            q = q.filter(Notification.read == False)
        notifications = q.order_by(Notification.created_at.desc()).all()
        unread_count = db.query(Notification).filter(
            Notification.user_id == current_user.id,
            Notification.read == False,
        ).count()

        result = [
            {
                "id": n.id,
                "type": n.type,
                "title": n.title,
                "message": n.message,
                "data": n.data,
                "read": n.read,
                "created_at": n.created_at.isoformat() if n.created_at else None,
            }
            for n in notifications
        ]

    return {"notifications": result, "unread_count": unread_count}


@app.post("/notifications/{notification_id}/read", tags=["Notifications"])
async def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
):
    with SessionLocal() as db:
        notif = db.query(Notification).filter(
            Notification.id == notification_id,
            Notification.user_id == current_user.id,
        ).first()
        if not notif:
            raise HTTPException(status_code=404, detail="Notification not found.")
        notif.read = True
        db.commit()
    return {"status": "ok"}


@app.post("/notifications/read-all", tags=["Notifications"])
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
):
    with SessionLocal() as db:
        result = db.query(Notification).filter(
            Notification.user_id == current_user.id,
            Notification.read == False,
        ).all()
        count = len(result)
        for n in result:
            n.read = True
        db.commit()
    return {"marked": count}


# ─────────────────────────────────────────────
# INVITATIONS — Accept external invite
# ─────────────────────────────────────────────

@app.get("/invitations/{token}", tags=["Invitations"])
async def get_invitation(token: str):
    """
    Returns invitation context (process name, inviter, email) so the
    accept-invite page can show the user what they're joining.
    No auth required. Returns 404 if token not found, expired, or used.
    """
    with SessionLocal() as db:
        invitation = db.query(Invitation).filter(
            Invitation.token == token,
        ).first()
        if not invitation:
            raise HTTPException(status_code=404, detail="Invitation not found.")
        if invitation.status != "pending":
            raise HTTPException(status_code=404, detail="Invitation is no longer valid.")
        inv_expires = invitation.expires_at
        if inv_expires.tzinfo is None:
            inv_expires = inv_expires.replace(tzinfo=timezone.utc)
        if inv_expires < datetime.now(timezone.utc):
            invitation.status = "expired"
            db.commit()
            raise HTTPException(status_code=404, detail="Invitation has expired.")

        analysis = db.query(db_models.Analysis).filter(
            db_models.Analysis.id == invitation.analysis_id,
        ).first()
        inviter = db.query(User).filter(User.id == invitation.invited_by).first()

        return {
            "process_name":     analysis.process_name if analysis else "Unknown Process",
            "invited_by_name":  inviter.full_name if inviter else "Unknown",
            "email":            invitation.email,
            "expires_at":       invitation.expires_at.isoformat(),
        }


@app.get("/accept-invite/{token}", include_in_schema=False)
async def accept_invite_page(token: str):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/docs/accept-invite.html?token={token}")


class AcceptInviteRequest(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None
    full_name: Optional[str] = None
    existing: bool = False


@app.post("/invitations/{token}/accept", tags=["Invitations"])
async def accept_process_invitation(
    token: str,
    body: AcceptInviteRequest,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Accept a process collaboration invitation.
    If body.existing=True and current_user is authenticated, re-uses their account.
    Otherwise creates a new user account with business_role=colaborador.
    """
    from auth.password import hash_password as _hash
    from auth.jwt import create_access_token, create_refresh_token
    import hashlib

    with SessionLocal() as db:
        invitation = db.query(Invitation).filter(
            Invitation.token == token,
            Invitation.status == "pending",
        ).first()
        if not invitation:
            raise HTTPException(status_code=404, detail="Invitation not found or already used.")
        inv_expires = invitation.expires_at
        if inv_expires.tzinfo is None:
            inv_expires = inv_expires.replace(tzinfo=timezone.utc)
        if inv_expires < datetime.now(timezone.utc):
            invitation.status = "expired"
            db.commit()
            raise HTTPException(status_code=410, detail="Invitation has expired.")

        # Resolve which user accepts
        is_new_user = False
        if body.existing and current_user:
            user = current_user
        else:
            if not body.email or not body.password or not body.full_name:
                raise HTTPException(
                    status_code=422,
                    detail="email, password, and full_name are required for new users.",
                )
            if db.query(User).filter(User.email == body.email).first():
                raise HTTPException(status_code=409, detail="Email already registered.")

            # Find the analysis to get tenant
            analysis = db.query(db_models.Analysis).filter(
                db_models.Analysis.id == invitation.analysis_id,
            ).first()
            if not analysis or not analysis.tenant_id:
                raise HTTPException(status_code=400, detail="Analysis tenant not found.")

            user = User(
                email=body.email,
                hashed_password=_hash(body.password),
                full_name=body.full_name,
                role="member",
                business_role="colaborador",
                tenant_id=analysis.tenant_id,
            )
            db.add(user)
            db.flush()
            is_new_user = True

        # Create collaborator record (active — they accepted)
        existing_collab = db.query(ProcessCollaborator).filter(
            ProcessCollaborator.analysis_id == invitation.analysis_id,
            ProcessCollaborator.user_id == user.id,
        ).first()
        if not existing_collab:
            collab = ProcessCollaborator(
                analysis_id=invitation.analysis_id,
                user_id=user.id,
                invited_by=invitation.invited_by,
                status="active",
            )
            db.add(collab)
        else:
            existing_collab.status = "active"

        # Welcome notification for new colaboradores joining via external invite
        if is_new_user:
            inv_analysis = db.query(db_models.Analysis).filter(
                db_models.Analysis.id == invitation.analysis_id,
            ).first()
            process_name = inv_analysis.process_name if inv_analysis else "un proceso"
            db.add(Notification(
                user_id=user.id,
                type="welcome",
                title="Bienvenido a Processa AI",
                message=f"Has sido agregado como colaborador en {process_name}",
                data=json.dumps({"analysis_id": invitation.analysis_id}),
            ))

        invitation.status = "accepted"
        db.commit()
        db.refresh(user)

        # Issue tokens
        access = create_access_token(user.id, user.tenant_id, user.role)
        refresh, expires_at = create_refresh_token(user.id)
        token_hash = hashlib.sha256(refresh.encode()).hexdigest()
        db.add(db_models.RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=expires_at,
        ))
        db.commit()

        logger.info(f"Invite accepted: token={token} user={user.email} analysis={invitation.analysis_id}")
        from models.schemas import TokenResponse
        return TokenResponse(
            access_token=access,
            refresh_token=refresh,
            business_role=user.business_role or "colaborador",
        )


# ─────────────────────────────────────────────
# HITL TIMEOUT MONITOR
# ─────────────────────────────────────────────

HITL_TIMEOUT_HOURS = 24


async def hitl_timeout_monitor() -> None:
    """Background task: marks stale HITL sessions as timed_out every hour."""
    while True:
        await asyncio.sleep(3600)
        now = datetime.now(timezone.utc)
        for sid, state in list(_sessions.items()):
            asis_stale = (
                state.hitl_asis_required
                and not state.hitl_asis_approved
                and state.hitl_asis_started_at is not None
                and (now - state.hitl_asis_started_at).total_seconds() > HITL_TIMEOUT_HOURS * 3600
            )
            tobe_stale = (
                state.hitl_required
                and not state.hitl_approved
                and state.hitl_retries > 0
                # hitl_retries > 0 means the node was reached; use a proxy for started_at
            )
            if asis_stale or tobe_stale:
                state.current_node = "timed_out"
                state.errors.append("HITL timeout: sesión expirada sin revisión en 24h")
                _sessions[sid] = state
                logger.warning(f"HITL timeout: sesión {sid} marcada como timed_out")


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(hitl_timeout_monitor())
    logger.info("HITL timeout monitor iniciado")
