from __future__ import annotations
from langchain_core.documents import Document

from models.schemas import Process, WasteAnalysisResult
from rag.vector_store import retrieve_similar_cases, retrieve_lean_patterns
from observability.logger import get_logger

logger = get_logger(__name__)


def _build_process_query(process: Process, analysis: WasteAnalysisResult) -> str:
    """
    Construye una query semántica rica para el retrieval,
    combinando nombre, industria, desperdicios detectados y participantes.
    """
    waste_types = ", ".join(wt.value for wt in analysis.main_waste_types)
    participants = ", ".join(process.participants[:4])
    systems = ", ".join(process.systems[:4])

    query = (
        f"Proceso: {process.name}. "
        f"Área: {process.owner}. "
        f"Participantes: {participants}. "
        f"Sistemas: {systems}. "
        f"Desperdicios detectados: {waste_types}. "
        f"Descripción: {process.description[:200]}"
    )
    return query


def _build_lean_query(analysis: WasteAnalysisResult) -> str:
    """
    Query orientada a recuperar patrones Lean/Six Sigma
    específicos para los desperdicios detectados.
    """
    waste_types = " ".join(wt.value for wt in analysis.main_waste_types)
    automation_pct = analysis.automation_coverage_pct

    return (
        f"Técnicas Lean para eliminar {waste_types}. "
        f"Proceso con {analysis.waste_percentage:.0f}% de desperdicios. "
        f"{automation_pct:.0f}% de actividades automatizables. "
        f"Kaizen quick wins y Six Sigma DMAIC aplicados."
    )


def retrieve_context(
    process: Process,
    analysis: WasteAnalysisResult,
) -> list[str]:
    """
    Punto de entrada principal del RAG.
    Recupera contexto de dos fuentes y lo unifica como lista de strings.

    Returns:
        Lista de fragmentos de contexto listos para inyectar al prompt TO-BE.
    """
    process_query = _build_process_query(process, analysis)
    lean_query    = _build_lean_query(analysis)

    # Recuperación paralela de ambas colecciones
    similar_cases: list[Document] = retrieve_similar_cases(process_query)
    lean_patterns: list[Document] = retrieve_lean_patterns(lean_query)

    context_chunks: list[str] = []

    if similar_cases:
        context_chunks.append("=== CASOS DE PROCESO SIMILARES ===")
        for i, doc in enumerate(similar_cases, 1):
            context_chunks.append(
                f"[Caso {i}] {doc.metadata.get('process_name', 'Sin nombre')}\n"
                f"{doc.page_content[:600]}"
            )

    if lean_patterns:
        context_chunks.append("=== PATRONES LEAN / SIX SIGMA / KAIZEN ===")
        for i, doc in enumerate(lean_patterns, 1):
            context_chunks.append(
                f"[Patrón {i}]\n{doc.page_content[:600]}"
            )

    if not context_chunks:
        logger.warning("RAG: No se encontró contexto relevante — el optimizer usará solo el LLM base")
        context_chunks = ["No se encontraron casos similares previos. Aplica mejores prácticas generales de Lean y Six Sigma."]

    logger.info(
        f"RAG context construido: {len(similar_cases)} casos + "
        f"{len(lean_patterns)} patrones Lean"
    )
    return context_chunks


# ─────────────────────────────────────────────
# NODO LANGGRAPH
# ─────────────────────────────────────────────

def node_retrieve_rag(state) -> dict:
    """
    Nodo LangGraph: recupera contexto RAG para enriquecer la generación TO-BE.

    Entrada del estado: asis_process, waste_analysis
    Salida al estado:   rag_context, current_node
    """
    from models.schemas import AgentState
    logger.info("── Nodo: retrieve_rag ──")

    process  = state.asis_process
    analysis = state.waste_analysis

    if process is None or analysis is None:
        logger.warning("RAG: proceso o análisis faltantes — continuando sin contexto")
        return {
            "rag_context":   ["Sin contexto RAG disponible."],
            "current_node":  "retrieve_rag",
        }

    try:
        context = retrieve_context(process, analysis)
        return {
            "rag_context":  context,
            "current_node": "retrieve_rag",
        }
    except Exception as e:
        # El RAG no es bloqueante — si falla, continúa sin contexto
        logger.error(f"Error en RAG retrieval: {e} — continuando sin contexto")
        return {
            "rag_context":  [f"RAG no disponible: {str(e)}"],
            "current_node": "retrieve_rag",
        }