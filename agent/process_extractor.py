from __future__ import annotations
import json
from tenacity import retry, stop_after_attempt, wait_exponential

#from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
from models.schemas import AgentState, Process, Activity, SubActivity
from prompts.extract_asis import extract_asis_prompt
from observability.logger import get_logger
from llm.factory import get_llm

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# LLM + PARSER
# ─────────────────────────────────────────────

#def _build_llm() -> ChatOpenAI:
#    return ChatOpenAI(
#        model=settings.openai_model,
#        temperature=settings.openai_temperature,
#        api_key=settings.openai_api_key,
#    )
def _build_llm():
    return get_llm(temperature=settings.openai_temperature)


# ─────────────────────────────────────────────
# VALIDACIÓN Y CONSTRUCCIÓN DEL PROCESO
# ─────────────────────────────────────────────

def _build_process_from_dict(data: dict, raw_input: str) -> Process:
    """
    Construye y valida el objeto Process desde el dict retornado por el LLM.
    Aplica correcciones menores si el LLM omite campos opcionales.
    """
    activities: list[Activity] = []

    for act_data in data.get("activities", []):
        subactivities = [
            SubActivity(**sa) for sa in act_data.pop("subactivities", [])
        ]
        activity = Activity(**act_data, subactivities=subactivities)
        activities.append(activity)

    total_duration = sum(a.estimated_duration_min for a in activities)

    process = Process(
        id=data.get("id", "PROC-001"),
        name=data.get("name", "Proceso sin nombre"),
        description=data.get("description", ""),
        owner=data.get("owner", "No especificado"),
        scope=data.get("scope", "No especificado"),
        participants=data.get("participants", []),
        systems=data.get("systems", []),
        activities=activities,
        total_duration_min=data.get("total_duration_min", total_duration),
        raw_input=raw_input,
    )

    logger.info(
        f"Proceso extraído: '{process.name}' — "
        f"{len(activities)} actividades — "
        f"{process.total_duration_min:.1f} min totales"
    )
    return process


# ─────────────────────────────────────────────
# EXTRACCIÓN CON REINTENTOS
# ─────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
#def _call_llm_with_retry(llm: ChatOpenAI, raw_input: str) -> dict:
def _call_llm_with_retry(llm, raw_input: str) -> dict:
    """
    Llama al LLM con reintentos exponenciales.
    Parsea la respuesta JSON y valida estructura mínima.
    """
    json_schema = json.dumps(
        Process.model_json_schema(), ensure_ascii=False, indent=2
    )

    chain = extract_asis_prompt | llm | JsonOutputParser()

    result: dict = chain.invoke({
        "raw_input":   raw_input,
        "json_schema": json_schema,
    })

    # Validación mínima — debe tener al menos nombre y actividades
    if "name" not in result or "activities" not in result:
        raise ValueError(
            "La respuesta del LLM no contiene los campos mínimos requeridos "
            f"('name', 'activities'). Recibido: {list(result.keys())}"
        )

    return result


# ─────────────────────────────────────────────
# NODO LANGGRAPH
# ─────────────────────────────────────────────

def node_extract_asis(state: AgentState) -> dict:
    """
    Nodo LangGraph: extrae el proceso AS-IS estructurado desde el texto cargado.

    Entrada del estado: raw_input (texto del proceso)
    Salida al estado:   asis_process, extraction_ok, errors
    """
    logger.info("── Nodo: extract_asis ──")

    raw_input = state.raw_input
    if not raw_input.strip():
        return {
            "extraction_ok": False,
            "errors": state.errors + ["El campo raw_input está vacío."],
            "current_node": "extract_asis",
        }

    llm = _build_llm()

    try:
        data = _call_llm_with_retry(llm, raw_input)
        process = _build_process_from_dict(data, raw_input)

        return {
            "asis_process":   process,
            "extraction_ok":  True,
            "current_node":   "extract_asis",
        }

    except Exception as e:
        logger.error(f"Error en extracción AS-IS: {e}")
        return {
            "extraction_ok": False,
            "errors": state.errors + [f"extract_asis: {str(e)}"],
            "current_node": "extract_asis",
        }