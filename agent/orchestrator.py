# Esqueleto del orquestador del agente de procesos
from langgraph.graph import StateGraph, END
from models.schemas import AgentState
from config.settings import settings
from agent.document_loader import node_load_document   # ✅ implementado
from agent.process_extractor import node_extract_asis  # ✅ implementado
from agent.analyzer import node_analyze_waste   # ✅ implementado
from rag.retriever  import node_retrieve_rag    # ✅ implementado
from agent.optimizer import node_optimize_tobe  # ✅ implementado
from agent.optimizer import node_hitl_review    # ✅ implementado
from agent.bpmn_generator import node_generate_bpmn  # ✅ implementado
from agent.kpi_calculator import node_calculate_kpis   # ✅ implementado


# ── Nodos placeholder ────────────────────────────────────────────────────────
# Cada nodo recibe el estado completo y retorna un dict con los campos a actualizar.

#def node_load_document(state: AgentState) -> dict:
#    """Carga y parsea el documento de entrada (PDF, Excel, texto)."""
#    return {"current_node": "load_document"}


#def node_extract_asis(state: AgentState) -> dict:
#    """LLM extrae el proceso AS-IS estructurado."""
#    return {"current_node": "extract_asis"}


#def node_analyze_waste(state: AgentState) -> dict:
#    """Detecta Muda, redundancias y clasifica actividades."""
#    return {"current_node": "analyze_waste"}


#def node_retrieve_rag(state: AgentState) -> dict:
#    """RAG: recupera procesos similares y patrones Lean/Six Sigma."""
#    return {"current_node": "retrieve_rag"}


#def node_optimize_tobe(state: AgentState) -> dict:
#    """Genera la propuesta TO-BE optimizada."""
#    return {"current_node": "optimize_tobe"}


#def node_hitl_review(state: AgentState) -> dict:
#    """Pausa para validación humana del TO-BE antes de continuar."""
#    return {"current_node": "hitl_review", "hitl_required": True}


#def node_generate_bpmn(state: AgentState) -> dict:
#    """Genera el XML BPMN 2.0 del proceso TO-BE aprobado."""
#    return {"current_node": "generate_bpmn"}


#def node_calculate_kpis(state: AgentState) -> dict:
#    """Calcula métricas cuantitativas AS-IS vs TO-BE."""
#    return {"current_node": "calculate_kpis"}


# ── Condiciones de routing ───────────────────────────────────────────────────

def route_after_extraction(state: AgentState) -> str:
    if not state.extraction_ok:
        return "end"          # Extracción fallida → terminar con error
    return "analyze_waste"


def route_after_optimization(state: AgentState) -> str:
    if settings.hitl_enabled:
        return "hitl_review"  # HITL activo → pausa para aprobación
    return "generate_bpmn"    # HITL desactivado → continuar directo


def route_after_hitl(state: AgentState) -> str:
    if state.hitl_approved:
        return "generate_bpmn"
    # Máximo 2 re-optimizaciones para no desperdiciar tokens
    if state.hitl_retries >= 2:
        return "generate_bpmn"  # Fuerza continuar tras 2 intentos
    return "optimize_tobe"


# ── Construcción del grafo ───────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Registro de nodos
    graph.add_node("load_document",  node_load_document)
    graph.add_node("extract_asis",   node_extract_asis)
    graph.add_node("analyze_waste",  node_analyze_waste)
    graph.add_node("retrieve_rag",   node_retrieve_rag)
    graph.add_node("optimize_tobe",  node_optimize_tobe)
    graph.add_node("hitl_review",    node_hitl_review)
    graph.add_node("generate_bpmn",  node_generate_bpmn)
    graph.add_node("calculate_kpis", node_calculate_kpis)

    # Flujo principal
    graph.set_entry_point("load_document")
    graph.add_edge("load_document", "extract_asis")

    # Routing condicional post-extracción
    graph.add_conditional_edges(
        "extract_asis",
        route_after_extraction,
        {"analyze_waste": "analyze_waste", "end": END}
    )

    graph.add_edge("analyze_waste", "retrieve_rag")
    graph.add_edge("retrieve_rag",  "optimize_tobe")

    # Routing condicional post-optimización (HITL)
    graph.add_conditional_edges(
        "optimize_tobe",
        route_after_optimization,
        {"hitl_review": "hitl_review", "generate_bpmn": "generate_bpmn"}
    )

    # Routing condicional post-HITL
    graph.add_conditional_edges(
        "hitl_review",
        route_after_hitl,
        {"generate_bpmn": "generate_bpmn", "optimize_tobe": "optimize_tobe"}
    )

    graph.add_edge("generate_bpmn",  "calculate_kpis")
    graph.add_edge("calculate_kpis", END)

    return graph.compile()


# Instancia global del grafo compilado
optimizer_graph = build_graph()