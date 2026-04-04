from __future__ import annotations
import json
from collections import Counter
from tenacity import retry, stop_after_attempt, wait_exponential

#from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
from models.schemas import (
    AgentState,
    ActivityWasteDetail,
    Redundancy,
    WasteAnalysisResult,
    WasteClassification,
    WasteType,
)
from prompts.detect_muda import detect_muda_prompt
from observability.logger import get_logger
from llm.factory import get_llm

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

#def _build_llm() -> ChatOpenAI:
#    return ChatOpenAI(
#        model=settings.openai_model,
#        temperature=0.1,          # Baja temperatura para análisis determinístico
#        api_key=settings.openai_api_key,
#    )
def _build_llm():
    return get_llm(temperature=0.1)

# ─────────────────────────────────────────────
# CONSTRUCCIÓN DEL RESULTADO
# ─────────────────────────────────────────────

def _build_waste_analysis(
    data: dict,
    process_id: str,
    process_name: str,
) -> WasteAnalysisResult:
    """
    Construye y valida el WasteAnalysisResult desde el dict del LLM.
    Calcula automáticamente todos los totales y porcentajes.
    """
    # Construir detalles por actividad
    activity_details: list[ActivityWasteDetail] = []
    for item in data.get("activity_details", []):
        detail = ActivityWasteDetail(**item)
        activity_details.append(detail)

    # Construir redundancias
    redundancies: list[Redundancy] = [
        Redundancy(**r) for r in data.get("redundancies", [])
    ]

    # ── Calcular totales ───────────────────────────────────────────────────
    total = len(activity_details)
    value_added = sum(
        1 for a in activity_details
        if a.waste_classification == WasteClassification.VALUE_ADDED
    )
    waste_count = sum(
        1 for a in activity_details
        if a.waste_classification == WasteClassification.WASTE
    )
    needs_info = sum(
        1 for a in activity_details
        if a.waste_classification == WasteClassification.NEEDS_INFO
    )
    total_waste_time = sum(
        a.estimated_waste_time_min for a in activity_details
    )
    automatable = sum(1 for a in activity_details if a.is_automatable)

    waste_pct = round((waste_count / total * 100), 1) if total > 0 else 0.0
    automation_pct = round((automatable / total * 100), 1) if total > 0 else 0.0

    # ── Tipos de Muda predominantes (top 3) ───────────────────────────────
    waste_type_counts = Counter(
        a.waste_type for a in activity_details
        if a.waste_type is not None
    )
    main_waste_types = [wt for wt, _ in waste_type_counts.most_common(3)]

    result = WasteAnalysisResult(
        process_id=process_id,
        process_name=process_name,
        activity_details=activity_details,
        redundancies=redundancies,
        total_activities=total,
        value_added_count=value_added,
        waste_count=waste_count,
        needs_info_count=needs_info,
        waste_percentage=waste_pct,
        total_waste_time_min=total_waste_time,
        automatable_count=automatable,
        automation_coverage_pct=automation_pct,
        main_waste_types=main_waste_types,
        lean_summary=data.get("lean_summary", ""),
        six_sigma_insights=data.get("six_sigma_insights", ""),
        kaizen_quick_wins=data.get("kaizen_quick_wins", []),
    )

    logger.info(
        f"Análisis completado: {waste_count}/{total} desperdicios "
        f"({waste_pct}%) — {automatable} automatizables — "
        f"{len(redundancies)} redundancias"
    )
    return result


# ─────────────────────────────────────────────
# LLAMADA AL LLM CON REINTENTOS
# ─────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
#def _call_llm_with_retry(llm: ChatOpenAI, asis_process_json: str) -> dict:
def _call_llm_with_retry(llm, asis_process_json: str) -> dict:
    json_schema = json.dumps(
        WasteAnalysisResult.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    chain = detect_muda_prompt | llm | JsonOutputParser()

    result: dict = chain.invoke({
        "asis_process_json": asis_process_json,
        "json_schema":       json_schema,
    })

    # Validación mínima
    if "activity_details" not in result:
        raise ValueError(
            "La respuesta del LLM no contiene 'activity_details'. "
            f"Campos recibidos: {list(result.keys())}"
        )

    return result


# ─────────────────────────────────────────────
# ANÁLISIS DETERMINÍSTICO (sin LLM)
# ─────────────────────────────────────────────

def _detect_sequential_waits(asis_json: dict) -> list[str]:
    """
    Heurística local: detecta actividades consecutivas de un mismo
    responsable separadas por esperas implícitas (aprobaciones, revisiones).
    Complementa el análisis LLM con reglas determinísticas.
    """
    insights: list[str] = []
    activities = asis_json.get("activities", [])

    approval_keywords = {"aprobación", "revisión", "validación", "verificación", "autorización"}
    wait_keywords     = {"espera", "pendiente", "en cola", "días hábiles"}

    for act in activities:
        name_lower = act.get("name", "").lower()
        desc_lower = act.get("description", "").lower()
        combined   = name_lower + " " + desc_lower

        if any(kw in combined for kw in approval_keywords | wait_keywords):
            insights.append(
                f"[Heurística] '{act['name']}' contiene patrón de espera/aprobación "
                f"— candidata a Muda tipo 'espera' o 'sobreproceso'."
            )

    return insights


def _detect_duplicate_systems(asis_json: dict) -> list[str]:
    """
    Heurística local: detecta sistemas usados en múltiples actividades
    que podrían indicar doble captura o transferencias redundantes.
    """
    system_usage: dict[str, list[str]] = {}
    for act in asis_json.get("activities", []):
        for sys in act.get("systems_used", []):
            system_usage.setdefault(sys, []).append(act["name"])

    insights: list[str] = []
    for sys, acts in system_usage.items():
        if len(acts) > 2:
            insights.append(
                f"[Heurística] El sistema '{sys}' se usa en {len(acts)} actividades "
                f"({', '.join(acts[:3])}...) — posible doble captura de datos."
            )
    return insights


# ─────────────────────────────────────────────
# NODO LANGGRAPH
# ─────────────────────────────────────────────

def node_analyze_waste(state: AgentState) -> dict:
    """
    Nodo LangGraph: analiza el proceso AS-IS aplicando Lean, Six Sigma y Kaizen.

    Entrada del estado: asis_process (Process validado)
    Salida al estado:   waste_analysis (WasteAnalysisResult), analysis_ok, errors
    """
    logger.info("── Nodo: analyze_waste ──")

    process = state.asis_process
    if process is None:
        return {
            "analysis_ok": False,
            "errors": state.errors + ["analyze_waste: asis_process es None."],
            "current_node": "analyze_waste",
        }

    # Serializar el proceso AS-IS para el LLM
    asis_json = process.model_dump(mode="json")
    asis_process_json = json.dumps(asis_json, ensure_ascii=False, indent=2)

    # ── Análisis heurístico local (rápido, sin LLM) ───────────────────────
    heuristic_insights = (
        _detect_sequential_waits(asis_json) +
        _detect_duplicate_systems(asis_json)
    )
    if heuristic_insights:
        logger.info(f"Heurísticas detectadas: {len(heuristic_insights)}")
        for insight in heuristic_insights:
            logger.info(insight)

    # ── Análisis LLM ──────────────────────────────────────────────────────
    llm = _build_llm()

    try:
        data = _call_llm_with_retry(llm, asis_process_json)

        # Enriquecer quick_wins con insights heurísticos si los hay
        if heuristic_insights:
            existing_qw = data.get("kaizen_quick_wins", [])
            data["kaizen_quick_wins"] = existing_qw + heuristic_insights

        waste_analysis = _build_waste_analysis(
            data=data,
            process_id=process.id,
            process_name=process.name,
        )

        return {
            "waste_analysis": waste_analysis,
            "analysis_ok":    True,
            "current_node":   "analyze_waste",
        }

    except Exception as e:
        logger.error(f"Error en analyze_waste: {e}")
        return {
            "analysis_ok": False,
            "errors": state.errors + [f"analyze_waste: {str(e)}"],
            "current_node": "analyze_waste",
        }