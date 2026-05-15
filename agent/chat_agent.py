"""
Chat contextual post-análisis.
Recibe el mensaje del usuario + el JSON completo del análisis como contexto,
llama a Llama 3.3 70B vía Groq y responde como consultor Lean/Six Sigma.
Mantiene historial de conversación en memoria por sesión.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth.dependencies import get_current_user
from llm.factory import get_llm
from observability.logger import get_logger
from storage.models import User
from storage.database import SessionLocal
from storage import repository as repo

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    mensaje: str = Field(
        ...,
        min_length=1,
        description="Mensaje del usuario",
    )
    contexto_analisis: dict = Field(
        default_factory=dict,
        description="JSON completo del análisis de proceso (AS-IS, desperdicios, TO-BE, KPIs). Puede ser vacío para colaboradores.",
    )
    session_id: str = Field(
        default="default",
        description="Identificador de sesión para mantener historial de chat",
    )


class ChatResponse(BaseModel):
    respuesta: str
    session_id: str
    mensajes_en_historial: int


# ─────────────────────────────────────────────
# IN-MEMORY CHAT HISTORY  (por sesión)
# ─────────────────────────────────────────────

_chat_histories: dict[str, list[dict]] = {}

_TIME_HINTS    = (
    "minuto", "minutos", "min ", "min.", "hora", "horas", "día", "dias", "días",
    "segundos", "tarda", "demora", "lleva", " hr", "h ", "h.", "semana",
)
_TOOL_HINTS    = (
    "excel", "sap", "outlook", "correo", "email", "salesforce", "monday", "trello",
    "jira", "slack", "teams", "drive", "sharepoint", "google", "powerpoint",
    "word", "tableau", "power bi", "powerbi", "sistema", "plataforma", "software",
    "erp", "crm", "rpa", "bot ",
)
_PAIN_HINTS    = (
    "problema", "demora", "lento", "error", "manual", "tedioso", "complic",
    "duplic", "espera", "esper", "no funciona", "dificil", "difícil", "queja",
    "doble", "frustra", "cuello de botella",
)
_PEOPLE_HINTS  = (
    "jefe", "gerente", "supervisor", "director", "área", "equipo", "departamento",
    "cliente", "proveedor", "compañero", "compañera", "rrhh", "finanzas",
    "operaciones", "comercial", "ventas", "ti ", "soporte",
)


def _analyze_collected_info(conversation_history: list[dict]) -> dict:
    """
    Heuristic check of what the colaborador has already mentioned, so the next
    LLM turn can target the gaps. Conservative — false negatives push the
    interview to probe deeper, which is the desired behavior.
    """
    user_text = " ".join(
        (m.get("content") or "").lower()
        for m in conversation_history
        if m.get("role") == "user"
    )
    if not user_text:
        return {}

    activity_count = sum(user_text.count(v) for v in (" hago", " reviso", " envío", " envio", " genero", " preparo", " valido", " apruebo", " creo", " ingreso", "luego ", "después ", "despues ", "primero ", "después de"))
    return {
        "activities":  activity_count >= 2,
        "durations":   any(h in user_text for h in _TIME_HINTS),
        "tools":       any(h in user_text for h in _TOOL_HINTS),
        "pain_points": any(h in user_text for h in _PAIN_HINTS),
        "people":      any(h in user_text for h in _PEOPLE_HINTS),
    }


def build_colaborador_prompt(
    process_name: str,
    department: str,
    collaborator_name: str,
    conversation_history: list[dict],
) -> str:
    """Sprint 8 — dynamic prompt adapting to what the colaborador has already said."""
    collected = _analyze_collected_info(conversation_history)

    missing: list[str] = []
    if not collected.get("activities"):  missing.append("actividades específicas (paso a paso)")
    if not collected.get("durations"):   missing.append("tiempos reales de cada actividad")
    if not collected.get("tools"):       missing.append("herramientas y sistemas utilizados")
    if not collected.get("pain_points"): missing.append("problemas, errores o demoras frecuentes")
    if not collected.get("people"):      missing.append("personas o roles con quienes interactúa")

    missing_text = ", ".join(missing) if missing else "ninguno — ya tienes información completa"

    return (
        f"Eres un analista experto en levantamiento de procesos empresariales. "
        f"Estás entrevistando a {collaborator_name} sobre su participación en el proceso "
        f"\"{process_name}\" del área de {department}.\n\n"
        "Tu objetivo es extraer información COMPLETA y DETALLADA sobre:\n"
        "1. Cada actividad/tarea que realiza (paso a paso)\n"
        "2. Tiempo real que toma cada actividad\n"
        "3. Herramientas, sistemas o software que usa\n"
        "4. Personas o equipos con quienes interactúa\n"
        "5. Problemas, errores o demoras frecuentes\n"
        "6. Casos especiales o excepciones al proceso normal\n\n"
        f"INFORMACIÓN AÚN POR PROFUNDIZAR EN ESTA ENTREVISTA: {missing_text}\n\n"
        "INSTRUCCIONES:\n"
        "- Sé DINÁMICO: adapta tus preguntas a lo que ya te han contado.\n"
        "- Sé CRÍTICO: si algo no está claro, pide más detalles o ejemplos concretos.\n"
        "- Haz UNA pregunta a la vez. No abrumes con varias preguntas seguidas.\n"
        "- Si mencionan una herramienta, pregunta cómo la usan exactamente.\n"
        "- Si mencionan un tiempo, valida si es siempre así o varía.\n"
        "- Si algo suena a desperdicio o ineficiencia, profundiza.\n"
        "- Cuando tengas información completa de TODOS los puntos arriba, "
        "muestra un resumen estructurado y pregunta si está correcto antes de cerrar.\n\n"
        "Responde SIEMPRE en español. Tono amigable pero profesional, sin jerga técnica innecesaria."
    )


# Legacy export — kept for backward compatibility with code that imports the
# static template directly. New code paths use build_colaborador_prompt().
COLABORADOR_SYSTEM_PROMPT = (
    "Eres un asistente experto en levantamiento de procesos. "
    "Conversa con el colaborador sobre el proceso \"{process_name}\" "
    "y captura sus actividades, tiempos, herramientas y problemas. "
    "Responde siempre en español."
)


SYSTEM_PROMPT_TEMPLATE = """\
Eres un consultor experto en Lean Manufacturing, Six Sigma y Kaizen con 20+ años de experiencia \
optimizando procesos empresariales. El usuario acaba de recibir un análisis de su proceso \
y tiene preguntas de seguimiento.

Responde de forma concreta, práctica y accionable. Usa datos del análisis cuando sea relevante. \
Formatea con markdown cuando ayude a la legibilidad (listas, negritas, etc.). \
Responde siempre en español.

────────────────────────────────
CONTEXTO DEL ANÁLISIS DEL PROCESO:
{contexto}
────────────────────────────────

Instrucciones adicionales:
- Si te preguntan sobre tiempos de implementación, da rangos realistas basados en la complejidad del proceso.
- Si te preguntan sobre herramientas, recomienda opciones concretas (nombre de software, metodología) y explica por qué.
- Si te preguntan sobre ROI, basa tus cálculos en los KPIs del análisis cuando estén disponibles.
- Sé empático y profesional. No repitas el análisis completo; céntrate en responder la pregunta puntual.
- Si no tienes suficiente información para responder algo específico, dilo honestamente y sugiere qué datos adicionales necesitarías.
"""

MAX_HISTORY_MESSAGES = 20  # Máximo de mensajes por sesión (para no exceder token limits)


def _truncate_context(ctx: dict) -> str:
    """Build a concise context string from analysis dict, max ~1500 chars total."""
    parts = []

    asis = ctx.get("asis_process") or {}
    asis_name = asis.get("name", "N/A")
    asis_acts = ", ".join(
        (a.get("name") or str(a)) for a in (asis.get("activities") or [])
    )
    parts.append(f"AS-IS Process: {asis_name}. Activities: {asis_acts}"[:500])

    tobe = ctx.get("tobe_process") or {}
    tobe_name = tobe.get("name", "N/A")
    tobe_acts = ", ".join(
        (a.get("name") or str(a)) for a in (tobe.get("activities") or [])
    )
    parts.append(f"TO-BE Process: {tobe_name}. Activities: {tobe_acts}"[:500])

    kpi = ctx.get("kpi_report") or {}
    summary = kpi.get("executive_summary", "")
    if not summary:
        ct = kpi.get("cycle_time") or {}
        summary = (
            f"Cycle time -{ct.get('reduction_pct', '?')}%"
            f", ROI {kpi.get('estimated_roi_pct', '?')}%"
        )
    parts.append(f"KPIs: {summary}"[:200])

    waste = ctx.get("waste_analysis") or {}
    top = ", ".join(
        (w.get("waste_type") or "") for w in (waste.get("activity_details") or [])[:3]
    )
    parts.append(
        f"Waste: {waste.get('waste_percentage', '?')}%. Top wastes: {top}"[:200]
    )

    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat contextual post-análisis",
)
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """
    Recibe un mensaje del usuario junto con el contexto del análisis.
    Usa Llama 3.3 70B vía Groq para responder como consultor Lean/Six Sigma.
    Mantiene historial de conversación por sesión.
    """
    sid = request.session_id

    # Inicializar historial si no existe
    if sid not in _chat_histories:
        _chat_histories[sid] = []

    history = _chat_histories[sid]

    # Build system prompt — colaboradores receive a process-elicitation prompt
    # instead of the consultant follow-up prompt.
    if (current_user.business_role or "").lower() == "colaborador":
        process_name = "este proceso"
        department   = "—"
        analysis_id = sid.split("_")[0] if "_" in sid else sid
        try:
            with SessionLocal() as db:
                analysis = repo.get_analysis(db, analysis_id)
                if analysis:
                    if analysis.process_name:
                        process_name = analysis.process_name
                    if getattr(analysis, "department", None):
                        department = analysis.department
        except Exception as exc:
            logger.warning(f"No se pudo cargar process_name (sesión {sid}): {exc}")
        system_message = build_colaborador_prompt(
            process_name=process_name,
            department=department,
            collaborator_name=current_user.full_name or "el colaborador",
            conversation_history=history,
        )
    else:
        # Construir system prompt con contexto del análisis (truncado para evitar 413)
        contexto_str = _truncate_context(request.contexto_analisis)
        system_message = SYSTEM_PROMPT_TEMPLATE.format(contexto=contexto_str)

    # Construir lista de mensajes para el LLM
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages = [SystemMessage(content=system_message)]

    # Agregar historial previo (limitado)
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Agregar mensaje actual del usuario
    messages.append(HumanMessage(content=request.mensaje))

    try:
        llm = get_llm(temperature=0.4)
        response = llm.invoke(messages)
        respuesta_texto = response.content
    except Exception as e:
        logger.error(f"Error en chat LLM (sesión {sid}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar respuesta: {str(e)}",
        )

    # Guardar en historial en memoria
    history.append({"role": "user", "content": request.mensaje})
    history.append({"role": "assistant", "content": respuesta_texto})

    # Limitar tamaño del historial en memoria
    if len(history) > MAX_HISTORY_MESSAGES * 2:
        _chat_histories[sid] = history[-(MAX_HISTORY_MESSAGES * 2):]

    # Persistir en SQLite para supervivir reinicios
    try:
        with SessionLocal() as db:
            repo.save_chat_message(db, sid, "user", request.mensaje)
            repo.save_chat_message(db, sid, "assistant", respuesta_texto)
    except Exception as exc:
        logger.warning(f"No se pudo persistir chat en DB (sesión {sid}): {exc}")

    logger.info(f"Chat sesión {sid}: {len(history)} mensajes en historial")

    return ChatResponse(
        respuesta=respuesta_texto,
        session_id=sid,
        mensajes_en_historial=len(history),
    )


@router.delete(
    "/chat/{session_id}",
    tags=["Chat"],
    summary="Limpiar historial de chat",
)
async def clear_chat_history(session_id: str, current_user: User = Depends(get_current_user)):
    """Elimina el historial de chat de una sesión."""
    if session_id in _chat_histories:
        del _chat_histories[session_id]
    return {"message": f"Historial de chat para sesión '{session_id}' eliminado."}


@router.get(
    "/chat/{session_id}/history",
    tags=["Chat"],
    summary="Obtener historial de chat",
)
async def get_chat_history(session_id: str, current_user: User = Depends(get_current_user)):
    """Retorna el historial completo de chat de una sesión."""
    history = _chat_histories.get(session_id, [])
    if not history:
        try:
            with SessionLocal() as db:
                history = repo.get_chat_messages(db, session_id)
        except Exception as exc:
            logger.warning(f"No se pudo cargar chat desde DB (sesión {session_id}): {exc}")
    return {
        "session_id": session_id,
        "mensajes": history,
        "total": len(history),
    }
