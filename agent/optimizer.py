from __future__ import annotations
import json
from tenacity import retry, stop_after_attempt, wait_exponential

#from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
from models.schemas import (
    AgentState,
    TOBEProcess,
    OptimizedActivity,
    ActivityStatus,
    ActivityType,
)
from prompts.generate_tobe import generate_tobe_prompt
from observability.logger import get_logger
from llm.factory import get_llm

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

#def _build_llm() -> ChatOpenAI:
#    return ChatOpenAI(
#        model=settings.openai_model,
#        temperature=0.3,      # Algo de creatividad controlada para propuestas
#        api_key=settings.openai_api_key,
#    )
def _build_llm():
    return get_llm(temperature=0.3)

# ─────────────────────────────────────────────
# CONSTRUCCIÓN DEL TO-BE
# ─────────────────────────────────────────────

def _build_tobe_process(data: dict, original_process_id: str) -> TOBEProcess:
    """
    Construye y valida el TOBEProcess desde el dict del LLM.
    Calcula totales y valida coherencia de los campos.
    """
    activities: list[OptimizedActivity] = []

    for act_data in data.get("activities", []):
        # Normaliza status y type por si el LLM los retorna con variantes
        act_data["status"] = act_data.get("status", "conservada").lower()
        act_data["type"]   = act_data.get("type", "operativa").lower()

        # Actividades eliminadas tienen duración 0
        if act_data["status"] == ActivityStatus.ELIMINATED:
            act_data["estimated_duration_min"] = 0.0
            act_data["duration_reduction_pct"] = 100.0

        activities.append(OptimizedActivity(**act_data))

    # Duración total = suma de no-eliminadas
    total_duration = sum(
        a.estimated_duration_min for a in activities
        if a.status != ActivityStatus.ELIMINATED
    )

    tobe = TOBEProcess(
        id=data.get("id", f"TOBE-{original_process_id}"),
        original_process_id=original_process_id,
        name=data.get("name", "Proceso Optimizado TO-BE"),
        description=data.get("description", ""),
        owner=data.get("owner", ""),
        activities=activities,
        total_duration_min=total_duration,
        applied_methodologies=data.get(
            "applied_methodologies", ["Lean", "Six Sigma", "Kaizen"]
        ),
        sipoc=data.get("sipoc"),
        human_approved=False,
    )

    # Log de resumen
    status_counts = {}
    for a in activities:
        status_counts[a.status.value] = status_counts.get(a.status.value, 0) + 1

    logger.info(
        f"TO-BE generado: '{tobe.name}' — "
        f"{len(activities)} actividades — "
        f"{total_duration:.1f} min — "
        f"distribución: {status_counts}"
    )
    return tobe


# ─────────────────────────────────────────────
# LLAMADA LLM CON REINTENTOS
# ─────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm_with_retry(
    #llm: ChatOpenAI,
    llm,
    asis_json: str,
    waste_json: str,
    rag_context: str,
    hitl_feedback: str,
) -> dict:
    json_schema = json.dumps(
        TOBEProcess.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    chain = generate_tobe_prompt | llm | JsonOutputParser()

    result: dict = chain.invoke({
        "asis_process_json":  asis_json,
        "waste_analysis_json": waste_json,
        "rag_context":        rag_context,
        "hitl_feedback":      hitl_feedback,
        "json_schema":        json_schema,
    })

    if "activities" not in result or len(result["activities"]) == 0:
        raise ValueError(
            "El LLM no generó actividades TO-BE. "
            f"Campos recibidos: {list(result.keys())}"
        )

    return result


# ─────────────────────────────────────────────
# VALIDACIÓN POST-GENERACIÓN
# ─────────────────────────────────────────────

def _validate_tobe_coherence(tobe: TOBEProcess, asis_duration: float) -> list[str]:
    """
    Valida coherencia del TO-BE generado.
    Retorna lista de advertencias (no bloquea la ejecución).
    """
    warnings: list[str] = []

    # El TO-BE no debería ser más largo que el AS-IS
    if tobe.total_duration_min >= asis_duration:
        warnings.append(
            f"⚠️  El TO-BE ({tobe.total_duration_min:.1f} min) no es más corto "
            f"que el AS-IS ({asis_duration:.1f} min). Revisar optimizaciones."
        )

    # Debe haber al menos una actividad no conservada (algo debe cambiar)
    all_kept = all(
        a.status == ActivityStatus.KEPT for a in tobe.activities
    )
    if all_kept:
        warnings.append(
            "⚠️  Todas las actividades están 'conservadas' — "
            "el TO-BE no propone cambios reales."
        )

    # Actividades automatizadas deben tener herramienta especificada
    for act in tobe.activities:
        if act.status == ActivityStatus.AUTOMATED and not act.automation_tool:
            warnings.append(
                f"⚠️  Actividad '{act.name}' marcada como automatizada "
                f"pero sin herramienta especificada."
            )

    return warnings


# ─────────────────────────────────────────────
# NODO LANGGRAPH — OPTIMIZER
# ─────────────────────────────────────────────

def node_optimize_tobe(state: AgentState) -> dict:
    """
    Nodo LangGraph: genera la propuesta TO-BE optimizada.

    Entrada del estado: asis_process, waste_analysis, rag_context, hitl_feedback
    Salida al estado:   tobe_process, optimization_ok, errors
    """
    logger.info("── Nodo: optimize_tobe ──")

    process  = state.asis_process
    analysis = state.waste_analysis

    if process is None or analysis is None:
        return {
            "optimization_ok": False,
            "errors": state.errors + [
                "optimize_tobe: asis_process o waste_analysis son None."
            ],
            "current_node": "optimize_tobe",
        }

    # Serializar inputs para el LLM
    asis_json  = json.dumps(process.model_dump(mode="json"),   ensure_ascii=False, indent=2)
    waste_json = json.dumps(analysis.model_dump(mode="json"),  ensure_ascii=False, indent=2)
    rag_context = "\n\n".join(state.rag_context) if state.rag_context else "Sin contexto RAG."
    hitl_feedback = state.hitl_feedback or "Sin feedback del revisor."

    llm = _build_llm()

    try:
        data = _call_llm_with_retry(
            llm=llm,
            asis_json=asis_json,
            waste_json=waste_json,
            rag_context=rag_context,
            hitl_feedback=hitl_feedback,
        )

        tobe = _build_tobe_process(data, original_process_id=process.id)

        # Validar coherencia y loguear advertencias
        warnings = _validate_tobe_coherence(tobe, process.total_duration_min)
        for w in warnings:
            logger.warning(w)

        return {
            "tobe_process":    tobe,
            "optimization_ok": True,
            "current_node":    "optimize_tobe",
            "errors": state.errors + warnings,   # advertencias como errores suaves
        }

    except Exception as e:
        logger.error(f"Error en optimize_tobe: {e}")
        return {
            "optimization_ok": False,
            "errors": state.errors + [f"optimize_tobe: {str(e)}"],
            "current_node": "optimize_tobe",
        }


# ─────────────────────────────────────────────
# NODO LANGGRAPH — HITL REVIEW
# ─────────────────────────────────────────────

def node_hitl_review(state: AgentState) -> dict:
    """
    Nodo LangGraph: pausa para validación humana del TO-BE.

    En modo API (producción): el endpoint /review recibe la aprobación externa.
    En modo CLI (desarrollo): solicita input del usuario directamente.

    La aprobación se inyecta al estado mediante:
        state.hitl_approved = True/False
        state.hitl_feedback = "comentarios del revisor"
        state.tobe_process.human_approved = True
    """
    logger.info("── Nodo: hitl_review ──")

    if not settings.hitl_enabled:
        # HITL desactivado — auto-aprobar
        logger.info("HITL desactivado — aprobación automática")
        tobe = state.tobe_process
        if tobe:
            tobe.human_approved = True
        return {
            "hitl_approved":  True,
            "hitl_required":  False,
            "tobe_process":   tobe,
            "current_node":   "hitl_review",
        }

    # En modo producción, este nodo retorna el estado actual
    # y el grafo queda suspendido hasta que el endpoint /review
    # inyecte hitl_approved y reanude la ejecución.
    # LangGraph maneja esto con interrupt_before/interrupt_after.
    logger.info(
        "HITL activo — proceso suspendido esperando aprobación del revisor. "
        f"TO-BE: '{state.tobe_process.name if state.tobe_process else 'N/A'}'"
    )

    return {
        "hitl_required": True,
        "hitl_retries":  state.hitl_retries + 1,
        "current_node":  "hitl_review",
    }