import pytest
from unittest.mock import patch
from models.schemas import (
    AgentState, TOBEProcess, OptimizedActivity,
    ActivityStatus, ActivityType,
    WasteAnalysisResult, ActivityWasteDetail,
    WasteClassification, WasteType,
)
from agent.optimizer import (
    node_optimize_tobe,
    _build_tobe_process,
    _validate_tobe_coherence,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_TOBE_RESPONSE = {
    "id": "TOBE-PROC-001",
    "original_process_id": "PROC-001",
    "name": "Proceso de Compras Optimizado",
    "description": "Proceso rediseñado con Lean Six Sigma",
    "owner": "Área de Compras",
    "applied_methodologies": ["Lean", "Six Sigma", "Kaizen"],
    "sipoc": {
        "suppliers": ["Área solicitante"],
        "inputs": ["Solicitud de adquisición validada"],
        "process": ["Revisión y validación", "Búsqueda automática proveedores",
                    "Aprobación digital", "Generación orden de compra"],
        "outputs": ["Orden de compra en SAP"],
        "customers": ["Área solicitante", "Proveedor"],
    },
    "activities": [
        {
            "id": "OPT-001",
            "original_activity_id": "ACT-001",
            "name": "Revisión y validación integrada",
            "description": "Combina revisión de requisitos y validación en un solo paso",
            "responsible": "Analista de Compras",
            "type": "analitica",
            "status": "combinada",
            "estimated_duration_min": 25.0,
            "duration_reduction_pct": 40.0,
            "is_automatable": False,
            "automation_tool": None,
            "improvement_justification": "Eliminación de doble revisión (Muda sobreproceso)",
            "depends_on": [],
        },
        {
            "id": "OPT-002",
            "original_activity_id": "ACT-003",
            "name": "Búsqueda automática de proveedores",
            "description": "RPA consulta catálogo ERP automáticamente",
            "responsible": "Sistema RPA",
            "type": "operativa",
            "status": "automatizada",
            "estimated_duration_min": 5.0,
            "duration_reduction_pct": 88.0,
            "is_automatable": True,
            "automation_tool": "UiPath + ERP API",
            "improvement_justification": "Automatización RPA de tarea repetitiva estructurada",
            "depends_on": ["OPT-001"],
        },
        {
            "id": "OPT-003",
            "original_activity_id": "ACT-002",
            "name": "Aprobación digital automatizada",
            "description": "Flujo de aprobación por Power Automate con SLA de 2h",
            "responsible": "Jefe de Compras",
            "type": "cognitiva",
            "status": "optimizada",
            "estimated_duration_min": 120.0,
            "duration_reduction_pct": 95.0,
            "is_automatable": True,
            "automation_tool": "Power Automate",
            "improvement_justification": "Reducción de espera de 3 días a 2h (Muda espera)",
            "depends_on": ["OPT-002"],
        },
    ],
}


# ── Tests unitarios ───────────────────────────────────────────────────────────

def test_build_tobe_process_structure():
    tobe = _build_tobe_process(SAMPLE_TOBE_RESPONSE, "PROC-001")
    assert isinstance(tobe, TOBEProcess)
    assert tobe.original_process_id == "PROC-001"
    assert len(tobe.activities) == 3
    assert tobe.sipoc is not None


def test_build_tobe_process_total_duration():
    tobe = _build_tobe_process(SAMPLE_TOBE_RESPONSE, "PROC-001")
    # 25 + 5 + 120 = 150 min
    assert tobe.total_duration_min == 150.0


def test_build_tobe_eliminated_activity_duration():
    data = {**SAMPLE_TOBE_RESPONSE, "activities": [
        {**SAMPLE_TOBE_RESPONSE["activities"][0], "status": "eliminada",
         "estimated_duration_min": 999.0}
    ]}
    tobe = _build_tobe_process(data, "PROC-001")
    # Eliminadas no cuentan en el total
    assert tobe.total_duration_min == 0.0
    assert tobe.activities[0].estimated_duration_min == 0.0
    assert tobe.activities[0].duration_reduction_pct == 100.0


def test_validate_tobe_coherence_no_improvement():
    tobe = _build_tobe_process(SAMPLE_TOBE_RESPONSE, "PROC-001")
    tobe.total_duration_min = 3000.0   # Simula que es más largo que el AS-IS
    warnings = _validate_tobe_coherence(tobe, asis_duration=2975.0)
    assert any("no es más corto" in w for w in warnings)


def test_validate_tobe_coherence_all_kept():
    tobe = _build_tobe_process(SAMPLE_TOBE_RESPONSE, "PROC-001")
    for act in tobe.activities:
        act.status = ActivityStatus.KEPT
    warnings = _validate_tobe_coherence(tobe, asis_duration=2975.0)
    assert any("conservadas" in w for w in warnings)


def test_validate_tobe_automated_without_tool():
    tobe = _build_tobe_process(SAMPLE_TOBE_RESPONSE, "PROC-001")
    tobe.activities[1].automation_tool = None  # OPT-002 automatizada sin tool
    warnings = _validate_tobe_coherence(tobe, asis_duration=2975.0)
    assert any("sin herramienta" in w for w in warnings)


@patch("agent.optimizer._call_llm_with_retry")
def test_node_optimize_tobe_success(mock_llm, sample_agent_state):
    mock_llm.return_value = SAMPLE_TOBE_RESPONSE
    # Agregamos waste_analysis mínimo al estado
    from tests.test_analyzer import SAMPLE_LLM_RESPONSE
    from agent.analyzer import _build_waste_analysis
    sample_agent_state.waste_analysis = _build_waste_analysis(
        SAMPLE_LLM_RESPONSE, "PROC-001", "Proceso de Prueba"
    )

    result = node_optimize_tobe(sample_agent_state)

    assert result["optimization_ok"] is True
    assert result["tobe_process"] is not None
    assert len(result["tobe_process"].activities) == 3


@patch("agent.optimizer._call_llm_with_retry")
def test_node_optimize_tobe_llm_error(mock_llm, sample_agent_state):
    mock_llm.side_effect = Exception("API rate limit")
    from tests.test_analyzer import SAMPLE_LLM_RESPONSE
    from agent.analyzer import _build_waste_analysis
    sample_agent_state.waste_analysis = _build_waste_analysis(
        SAMPLE_LLM_RESPONSE, "PROC-001", "Proceso de Prueba"
    )

    result = node_optimize_tobe(sample_agent_state)

    assert result["optimization_ok"] is False
    assert "optimize_tobe" in result["errors"][-1]


def test_node_optimize_tobe_missing_process():
    state = AgentState(raw_input="...", asis_process=None, waste_analysis=None)
    result = node_optimize_tobe(state)
    assert result["optimization_ok"] is False


# ── Test de integración ───────────────────────────────────────────────────────

@pytest.mark.integration
def test_full_optimization_integration(sample_agent_state):
    """
    Test de integración real contra GPT-4o + Chroma.
    Ejecutar con: pytest -m integration
    """
    from tests.test_analyzer import SAMPLE_LLM_RESPONSE
    from agent.analyzer import _build_waste_analysis
    sample_agent_state.waste_analysis = _build_waste_analysis(
        SAMPLE_LLM_RESPONSE, "PROC-001", "Proceso de Prueba"
    )
    sample_agent_state.rag_context = ["Sin contexto RAG en test de integración."]

    result = node_optimize_tobe(sample_agent_state)

    assert result["optimization_ok"] is True
    tobe = result["tobe_process"]
    assert tobe.total_duration_min < sample_agent_state.asis_process.total_duration_min
    assert tobe.sipoc is not None
    assert len(tobe.activities) >= 2