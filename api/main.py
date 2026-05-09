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
# ANÁLISIS — TEXTO LIBRE
# ─────────────────────────────────────────────

@app.post(
    "/analyze/text",
    response_model=SessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Análisis"],
    summary="Analizar proceso desde texto libre",
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
    if recovered:
        with SessionLocal() as db:
            record = repo.get_analysis(db, session_id, user_id=current_user.id)
        if record:
            process_name = record.process_name

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
                "bpmn_file":      None,
            }

    return {
        "session_id":     session_id,
        "asis_process":   state.asis_process.model_dump() if state.asis_process else None,
        "waste_analysis": state.waste_analysis.model_dump() if state.waste_analysis else None,
        "tobe_process":   state.tobe_process.model_dump() if state.tobe_process else None,
        "kpi_report":     state.kpi_report.model_dump() if state.kpi_report else None,
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
    with SessionLocal() as db:
        # Analyses owned by user
        owned = repo.list_analyses(db, limit=limit, offset=offset, user_id=current_user.id)

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

        owned_ids = {r.id for r in owned}
        collab_map: dict = {}
        for analysis, collab in collaborated_query:
            if analysis.id not in owned_ids:
                collab_map[analysis.id] = (analysis, collab.status)

    def _fmt(r, is_owner: bool, collaborator_status=None):
        return {
            "id":                       r.id,
            "process_name":             r.process_name,
            "status":                   r.status,
            "score":                    r.score,
            "cycle_time_reduction_pct": r.cycle_time_reduction_pct,
            "automation_coverage_pct":  r.automation_coverage_pct,
            "created_at":               r.created_at.isoformat() if r.created_at else None,
            "completed_at":             r.completed_at.isoformat() if r.completed_at else None,
            "has_errors":               r.has_errors,
            "is_owner":                 is_owner,
            "collaborator_status":      collaborator_status,
        }

    analyses = [_fmt(r, is_owner=True) for r in owned]
    analyses += [_fmt(r, is_owner=False, collaborator_status=status) for r, status in collab_map.values()]

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

@app.post("/sessions/{session_id}/sources", tags=["Sources"])
async def upload_source(
    session_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Upload an additional source document to an active session's RAG context."""
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)

    sources_dir = Path(f"storage/outputs/sources/{session_id}")
    sources_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "upload"
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
