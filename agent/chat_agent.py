"""
Chat contextual post-análisis.
Recibe el mensaje del usuario + el JSON completo del análisis como contexto,
llama a Llama 3.3 70B vía Groq y responde como consultor Lean/Six Sigma.
Mantiene historial de conversación en memoria por sesión.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from llm.factory import get_llm
from observability.logger import get_logger

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
        ...,
        description="JSON completo del análisis de proceso (AS-IS, desperdicios, TO-BE, KPIs)",
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


# ─────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat contextual post-análisis",
)
async def chat(request: ChatRequest):
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

    # Construir system prompt con contexto del análisis
    import json
    contexto_str = json.dumps(request.contexto_analisis, indent=2, ensure_ascii=False)
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

    # Guardar en historial
    history.append({"role": "user", "content": request.mensaje})
    history.append({"role": "assistant", "content": respuesta_texto})

    # Limitar tamaño del historial
    if len(history) > MAX_HISTORY_MESSAGES * 2:
        _chat_histories[sid] = history[-(MAX_HISTORY_MESSAGES * 2):]

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
async def clear_chat_history(session_id: str):
    """Elimina el historial de chat de una sesión."""
    if session_id in _chat_histories:
        del _chat_histories[session_id]
    return {"message": f"Historial de chat para sesión '{session_id}' eliminado."}


@router.get(
    "/chat/{session_id}/history",
    tags=["Chat"],
    summary="Obtener historial de chat",
)
async def get_chat_history(session_id: str):
    """Retorna el historial completo de chat de una sesión."""
    history = _chat_histories.get(session_id, [])
    return {
        "session_id": session_id,
        "mensajes": history,
        "total": len(history),
    }
