from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.orchestrator import optimizer_graph, build_graph
from agent.document_loader import load_document
from agent.chat_agent import router as chat_router
from models.schemas import (
    AgentState,
    KPIReportV2,
    TOBEProcess,
    WasteAnalysisResult,
    BPMNOutput,
)
from config.settings import settings
from observability.logger import get_logger
from storage.database import engine, SessionLocal
from storage import models as db_models
from storage import repository as repo
db_models.Base.metadata.create_all(bind=engine)

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
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chat contextual router
app.include_router(chat_router)

# ─────────────────────────────────────────────
# STORE EN MEMORIA (reemplazar por Redis en producción)
# ─────────────────────────────────────────────

_sessions: dict[str, AgentState] = {}


def _get_session(session_id: str) -> AgentState:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sesión '{session_id}' no encontrada.",
        )
    return _sessions[session_id]


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
    approved: bool  = Field(..., description="True = aprobado, False = requiere cambios")
    feedback: str   = Field(
        default="",
        description="Comentarios del revisor (requerido si approved=False)",
    )


class SessionResponse(BaseModel):
    session_id:   str
    message:      str
    current_node: str
    status:       str


class AnalysisStatusResponse(BaseModel):
    session_id:      str
    current_node:    str
    extraction_ok:   bool
    analysis_ok:     bool
    optimization_ok: bool
    hitl_required:   bool
    hitl_approved:   bool
    bpmn_ok:         bool
    kpi_ok:          bool
    errors:          list[str]


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
async def health_check():
    """Verifica que el servicio esté activo."""
    return {
        "status":   "healthy",
        "version":  "1.0.0",
        "hitl":     settings.hitl_enabled,
        "model":    settings.openai_model,
    }


@app.get("/health/rag", tags=["Sistema"])
async def health_rag():
    """Verifica el estado de la Vector DB."""
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
):
    """
    Inicia el análisis de un proceso AS-IS descrito en lenguaje natural.
    Retorna un session_id para consultar el progreso y los resultados.
    """
    session_id = str(uuid.uuid4())
    state = AgentState(
        raw_input=request.raw_input,
        current_node="start",
    )
    _sessions[session_id] = state

    # Persistir en BD
    with SessionLocal() as db:
        repo.create_analysis(db, session_id, request.process_name, request.raw_input)

    logger.info(f"Nueva sesión: {session_id} — input: {len(request.raw_input)} chars")

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
):
    """
    Acepta un archivo PDF, Excel (.xlsx) o texto (.txt) con la descripción del proceso.
    """
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

    # Guardar temporalmente
    tmp_dir  = Path("/tmp/process_optimizer")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    tmp_path.write_bytes(await file.read())

    # Extraer texto ya en este punto para validar contenido
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
    )
    _sessions[session_id] = state

    logger.info(
        f"Nueva sesión desde archivo: {session_id} — "
        f"{file.filename} ({len(raw_text)} chars)"
    )

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

        # ✅ Guardar solo si invoke retornó resultado válido
        if result and isinstance(result, dict):
            _sessions[session_id] = AgentState(**result)
            # Persistir resultado completo en BD
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

        logger.info(
            f"Pipeline completado: {session_id} — "
            f"nodo final: {_sessions[session_id].current_node}"
        )

    except Exception as e:
        logger.error(f"Error en pipeline {session_id}: {e}")
        # ✅ Verificar que la sesión aún existe antes de mutarla
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
    summary="Consultar estado del análisis",
)
async def get_session_status(session_id: str):
    """
    Retorna el estado actual del pipeline para una sesión.
    Úsalo para polling hasta que kpi_ok=true o errors esté poblado.
    """
    state = _get_session(session_id)
    return AnalysisStatusResponse(
        session_id=session_id,
        current_node=state.current_node,
        extraction_ok=state.extraction_ok,
        analysis_ok=state.analysis_ok,
        optimization_ok=state.optimization_ok,
        hitl_required=state.hitl_required,
        hitl_approved=state.hitl_approved,
        bpmn_ok=state.bpmn_ok,
        kpi_ok=state.kpi_ok,
        errors=state.errors,
    )


@app.get(
    "/sessions/{session_id}/process",
    tags=["Resultados"],
    summary="Obtener proceso AS-IS extraído",
)
async def get_asis_process(session_id: str):
    """Retorna el proceso AS-IS estructurado extraído del input."""
    state = _get_session(session_id)
    if not state.extraction_ok or state.asis_process is None:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="El proceso AS-IS aún no está disponible.",
        )
    return state.asis_process.model_dump()


@app.get(
    "/sessions/{session_id}/analysis",
    tags=["Resultados"],
    summary="Obtener análisis de desperdicios Lean",
)
async def get_waste_analysis(session_id: str):
    """Retorna el análisis completo de Muda, redundancias y oportunidades."""
    state = _get_session(session_id)
    if not state.analysis_ok or state.waste_analysis is None:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="El análisis Lean aún no está disponible.",
        )
    return state.waste_analysis.model_dump()


@app.get(
    "/sessions/{session_id}/tobe",
    tags=["Resultados"],
    summary="Obtener propuesta TO-BE optimizada",
)
async def get_tobe_process(session_id: str):
    """Retorna la propuesta de proceso optimizado TO-BE."""
    state = _get_session(session_id)
    if not state.optimization_ok or state.tobe_process is None:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="La propuesta TO-BE aún no está disponible.",
        )
    return state.tobe_process.model_dump()


@app.get(
    "/sessions/{session_id}/kpis",
    tags=["Resultados"],
    summary="Obtener reporte de KPIs",
)
async def get_kpi_report(session_id: str):
    """Retorna el reporte completo de KPIs AS-IS vs TO-BE con ROI y nivel Sigma."""
    state = _get_session(session_id)
    if not state.kpi_ok or state.kpi_report is None:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="El reporte de KPIs aún no está disponible.",
        )
    return state.kpi_report.model_dump()


@app.get(
    "/sessions/{session_id}/bpmn",
    tags=["Resultados"],
    summary="Descargar diagrama BPMN 2.0",
    response_class=FileResponse,
)
async def download_bpmn(session_id: str):
    """Descarga el archivo .bpmn XML 2.0 del proceso TO-BE."""
    state = _get_session(session_id)
    if not state.bpmn_ok or state.bpmn_output is None:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="El diagrama BPMN aún no está disponible.",
        )
    file_path = Path(state.bpmn_output.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Archivo BPMN no encontrado en el servidor.",
        )
    return FileResponse(
        path=str(file_path),
        media_type="application/xml",
        filename=file_path.name,
    )


@app.get(
    "/sessions/{session_id}/report",
    tags=["Resultados"],
    summary="Obtener reporte completo del análisis",
)
async def get_full_report(session_id: str):
    """
    Retorna el reporte completo: AS-IS + análisis + TO-BE + KPIs en un solo response.
    Solo disponible cuando kpi_ok=true.
    """
    state = _get_session(session_id)
    if not state.kpi_ok:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="El análisis completo aún no está disponible. "
                   "Consulta /sessions/{session_id}/status para ver el progreso.",
        )
    return {
        "session_id":    session_id,
        "asis_process":  state.asis_process.model_dump() if state.asis_process else None,
        "waste_analysis": state.waste_analysis.model_dump() if state.waste_analysis else None,
        "tobe_process":  state.tobe_process.model_dump() if state.tobe_process else None,
        "kpi_report":    state.kpi_report.model_dump() if state.kpi_report else None,
        "bpmn_file":     state.bpmn_output.file_path if state.bpmn_output else None,
    }


# ─────────────────────────────────────────────
# HITL — REVISIÓN HUMANA
# ─────────────────────────────────────────────

@app.post(
    "/sessions/{session_id}/review",
    response_model=SessionResponse,
    tags=["HITL"],
    summary="Aprobar o rechazar la propuesta TO-BE",
)
async def submit_hitl_review(
    session_id: str,
    review: HITLReviewRequest,
    background_tasks: BackgroundTasks,
):
    """
    Endpoint de revisión humana (Human-in-the-Loop).
    - approved=true: continúa con la generación BPMN y KPIs
    - approved=false: re-optimiza con el feedback del revisor
    """
    state = _get_session(session_id)

    if not state.hitl_required:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Esta sesión no tiene una revisión humana pendiente.",
        )

    if not review.approved and not review.feedback.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Si el TO-BE es rechazado, debes proporcionar feedback para re-optimizar.",
        )

    # Inyectar decisión del revisor al estado
    state.hitl_approved  = review.approved
    state.hitl_feedback  = review.feedback or None
    state.hitl_required  = False

    if review.approved and state.tobe_process:
        state.tobe_process.human_approved  = True
        state.tobe_process.approver_notes  = review.feedback or None

    _sessions[session_id] = state

    # Reanudar el grafo desde el nodo hitl_review
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
    """
    Reanuda el grafo LangGraph después de la aprobación/rechazo HITL.
    LangGraph usa el thread_id para restaurar el estado del checkpoint.
    """
    state = _sessions[session_id]
    try:
        result = optimizer_graph.invoke(
            state.model_dump(),
            config={"configurable": {"thread_id": session_id}},
        )
        _sessions[session_id] = AgentState(**result)
        logger.info(f"Pipeline reanudado post-HITL: {session_id}")
    except Exception as e:
        logger.error(f"Error al reanudar pipeline {session_id}: {e}")
        _sessions[session_id].errors.append(f"resume_pipeline: {str(e)}")


# ─────────────────────────────────────────────
# RAG — ADMINISTRACIÓN
# ─────────────────────────────────────────────

@app.post(
    "/rag/seed",
    tags=["RAG"],
    summary="Inicializar knowledge base Lean/Six Sigma",
)
async def seed_knowledge_base():
    """
    Inicializa la knowledge base con patrones Lean, Six Sigma y Kaizen.
    Ejecutar UNA SOLA VEZ al desplegar el sistema por primera vez.
    """
    try:
        from rag.seed_knowledge import seed
        seed()
        from rag.vector_store import get_collection_stats
        stats = get_collection_stats()
        return {
            "message": "Knowledge base inicializada correctamente.",
            "stats":   stats,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al inicializar knowledge base: {str(e)}",
        )


@app.get(
    "/rag/stats",
    tags=["RAG"],
    summary="Estadísticas de la Vector DB",
)
async def rag_stats():
    """Retorna el número de documentos indexados en cada colección."""
    try:
        from rag.vector_store import get_collection_stats
        return get_collection_stats()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# HISTORIAL — ANÁLISIS PERSISTIDOS
# ─────────────────────────────────────────────

@app.get(
    "/analyses",
    tags=["Historial"],
    summary="Listar historial de análisis persistidos",
)
async def list_analyses(limit: int = 50, offset: int = 0):
    """Retorna el historial de análisis guardados en BD, ordenados por fecha descendente."""
    with SessionLocal() as db:
        records = repo.list_analyses(db, limit=limit, offset=offset)
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


@app.get(
    "/analyses/{session_id}",
    tags=["Historial"],
    summary="Obtener análisis completo desde BD",
)
async def get_analysis(session_id: str):
    """Retorna el resultado completo de un análisis persistido en BD."""
    with SessionLocal() as db:
        record = repo.get_analysis(db, session_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Análisis '{session_id}' no encontrado.")
    import json
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

@app.delete(
    "/sessions/{session_id}",
    tags=["Sesiones"],
    summary="Eliminar sesión",
)
async def delete_session(session_id: str):
    """Elimina una sesión y libera memoria."""
    _get_session(session_id)   # Valida que existe
    del _sessions[session_id]
    return {"message": f"Sesión '{session_id}' eliminada."}


@app.get(
    "/sessions",
    tags=["Sesiones"],
    summary="Listar sesiones activas",
)
async def list_sessions():
    """Lista todas las sesiones activas con su estado actual."""
    return {
        "total": len(_sessions),
        "sessions": [
            {
                "session_id":   sid,
                "current_node": s.current_node,
                "kpi_ok":       s.kpi_ok,
                "errors":       len(s.errors),
            }
            for sid, s in _sessions.items()
        ],
    }