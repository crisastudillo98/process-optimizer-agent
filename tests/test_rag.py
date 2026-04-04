"""
Tests del pipeline RAG: embedder, vector store y retriever.
Los tests de integración requieren ChromaDB corriendo.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from rag.retriever import (
    _build_process_query,
    _build_lean_query,
    retrieve_context,
)
from models.schemas import WasteType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_waste_analysis_with_types(full_analysis):
    from models.schemas import WasteType
    full_analysis.main_waste_types = [WasteType.WAITING, WasteType.OVERPROCESSING]
    full_analysis.waste_percentage = 50.0
    full_analysis.automation_coverage_pct = 75.0
    return full_analysis


# ── Tests de construcción de queries ─────────────────────────────────────────

def test_build_process_query(sample_process, sample_waste_analysis_with_types):
    query = _build_process_query(sample_process, sample_waste_analysis_with_types)
    assert sample_process.name in query
    assert sample_process.owner in query
    assert "espera" in query or "sobreproceso" in query


def test_build_lean_query(sample_waste_analysis_with_types):
    query = _build_lean_query(sample_waste_analysis_with_types)
    assert "espera" in query or "sobreproceso" in query
    assert "%" in query


def test_build_process_query_truncates_description(sample_process, full_analysis):
    sample_process.description = "X" * 500    # Descripción muy larga
    query = _build_process_query(sample_process, full_analysis)
    # La descripción se trunca a 200 chars en la query
    assert len(query) < 600


# ── Tests de retrieval (con mock) ─────────────────────────────────────────────

@patch("rag.retriever.retrieve_similar_cases")
@patch("rag.retriever.retrieve_lean_patterns")
def test_retrieve_context_with_results(
    mock_lean, mock_cases, sample_process, sample_waste_analysis_with_types
):
    mock_cases.return_value = [
        Document(
            page_content="Caso similar de facturación con esperas eliminadas",
            metadata={"process_name": "Facturación"},
        )
    ]
    mock_lean.return_value = [
        Document(
            page_content="Técnica Lean para eliminar esperas: flujo digital",
            metadata={"waste_type": "espera"},
        )
    ]

    context = retrieve_context(sample_process, sample_waste_analysis_with_types)

    assert len(context) > 0
    assert any("CASOS DE PROCESO SIMILARES" in c for c in context)
    assert any("PATRONES LEAN" in c for c in context)
    assert any("Facturación" in c for c in context)


@patch("rag.retriever.retrieve_similar_cases")
@patch("rag.retriever.retrieve_lean_patterns")
def test_retrieve_context_empty_results(
    mock_lean, mock_cases, sample_process, full_analysis
):
    mock_cases.return_value = []
    mock_lean.return_value = []

    context = retrieve_context(sample_process, full_analysis)

    # Debe retornar fallback — nunca lista vacía
    assert len(context) > 0
    assert any("No se encontraron" in c for c in context)


@patch("rag.retriever.retrieve_similar_cases")
@patch("rag.retriever.retrieve_lean_patterns")
def test_retrieve_context_rag_failure_non_blocking(
    mock_lean, mock_cases, sample_process, full_analysis
):
    mock_cases.side_effect = Exception("ChromaDB no disponible")
    mock_lean.side_effect = Exception("ChromaDB no disponible")

    from rag.retriever import node_retrieve_rag
    from models.schemas import AgentState

    state = AgentState(
        raw_input="...",
        asis_process=sample_process,
        waste_analysis=full_analysis,
    )
    result = node_retrieve_rag(state)

    # No debe propagar la excepción — retorna contexto vacío
    assert "rag_context" in result
    assert len(result["rag_context"]) > 0


# ── Tests de integración RAG ──────────────────────────────────────────────────

@pytest.mark.integration
def test_seed_and_retrieve_lean_patterns():
    """
    Inicializa la KB y verifica que los patrones son recuperables.
    Requiere ChromaDB activo.
    """
    from rag.seed_knowledge import seed
    from rag.vector_store import retrieve_lean_patterns

    seed()

    results = retrieve_lean_patterns("técnicas para eliminar esperas en procesos")
    assert len(results) > 0
    assert any("espera" in doc.page_content.lower() for doc in results)


@pytest.mark.integration
def test_store_and_retrieve_process_case(sample_process):
    """
    Persiste un caso y lo recupera por similitud semántica.
    Requiere ChromaDB activo.
    """
    from rag.vector_store import store_process_case, retrieve_similar_cases

    store_process_case(
        process_name=sample_process.name,
        asis_summary="Proceso manual con muchas esperas en aprobaciones",
        tobe_summary="Proceso digital con aprobaciones automáticas por Power Automate",
        industry="Finanzas",
        improvements=["Reducción 70% tiempo ciclo", "Automatización aprobaciones"],
    )

    results = retrieve_similar_cases(
        "proceso de aprobación manual con esperas largas en finanzas"
    )
    assert len(results) > 0
    assert any(sample_process.name in doc.page_content for doc in results)