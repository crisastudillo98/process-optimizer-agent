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
from auth.dependencies import get_current_user, require_admin
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
from storage.models import User
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
            full_report = {
                "asis_process":   final_state.asis_process.model_dump() if final_state.asis_process else None,
                "waste_analysis": final_state.waste_analysis.model_dump() if final_state.waste_analysis else None,
                "tobe_process":   final_state.tobe_process.model_dump() if final_state.tobe_process else None,
                "kpi_report":     final_state.kpi_report.model_dump() if final_state.kpi_report else None,
            }
            score = final_state.waste_analysis.waste_percentage if final_state.waste_analysis else None
            with SessionLocal() as db:
                repo.complete_analysis(db, session_id, full_report, score)
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
    if not state.bpmn_ok or state.bpmn_output is None:
        raise HTTPException(status_code=425, detail="El diagrama BPMN aún no está disponible.")
    file_path = Path(state.bpmn_output.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Archivo BPMN no encontrado en el servidor.")
    return FileResponse(path=str(file_path), media_type="application/xml", filename=file_path.name)


@app.get("/sessions/{session_id}/bpmn/asis", tags=["Resultados"], response_class=FileResponse)
async def download_asis_bpmn(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id, user_id=current_user.id)
    _assert_session_owner(state, current_user)
    if not state.bpmn_asis_output:
        raise HTTPException(status_code=425, detail="El diagrama BPMN AS-IS aún no está disponible.")
    file_path = Path(state.bpmn_asis_output)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Archivo BPMN AS-IS no encontrado en el servidor.")
    return FileResponse(path=str(file_path), media_type="application/xml", filename=file_path.name)


@app.get("/sessions/{session_id}/report", tags=["Resultados"])
async def get_full_report(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    state = _get_session(session_id)
    _assert_session_owner(state, current_user)
    if not state.kpi_ok:
        raise HTTPException(
            status_code=425,
            detail="El análisis completo aún no está disponible.",
        )
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
            # TO-BE HITL resume: graph can re-run because hitl_asis_approved=True
            # guards the asis hitl node; route_after_asis_hitl sends to retrieve_rag
            result = optimizer_graph.invoke(
                state.model_dump(),
                config={"configurable": {"thread_id": session_id}},
            )
            if result and isinstance(result, dict):
                current = result
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
        records = repo.list_analyses(db, limit=limit, offset=offset, user_id=current_user.id)
    return {
        "total": len(records),
        "analyses": [
            {
                "id":                       r.id,
                "process_name":             r.process_name,
                "status":                   r.status,
                "score":                    r.score,
                "cycle_time_reduction_pct": r.cycle_time_reduction_pct,
                "automation_coverage_pct":  r.automation_coverage_pct,
                "created_at":               r.created_at.isoformat() if r.created_at else None,
                "completed_at":             r.completed_at.isoformat() if r.completed_at else None,
                "has_errors":               r.has_errors,
            }
            for r in records
        ],
    }


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
