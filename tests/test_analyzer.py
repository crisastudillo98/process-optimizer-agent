import pytest
from unittest.mock import patch
from models.schemas import (
    AgentState, Process, Activity, WasteAnalysisResult,
    WasteClassification, WasteType, ActivityType,
)
from agent.analyzer import (
    node_analyze_waste,
    _build_waste_analysis,
    _detect_sequential_waits,
    _detect_duplicate_systems,
)
from datetime import datetime


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_activity(id: str, name: str, responsible: str = "Analista",
                  duration: float = 30.0, systems: list = None) -> Activity:
    return Activity(
        id=id,
        name=name,
        description=f"Descripción de {name}",
        responsible=responsible,
        type=ActivityType.ANALYTICAL,
        estimated_duration_min=duration,
        depends_on=[],
        systems_used=systems or [],
    )


def make_process() -> Process:
    return Process(
        id="PROC-001",
        name="Proceso de Compras",
        description="Proceso de adquisición de materiales",
        owner="Compras",
        scope="Solicitud → Orden de compra",
        participants=["Analista", "Jefe"],
        systems=["ERP", "SAP"],
        activities=[
            make_activity("ACT-001", "Revisión de solicitud", duration=30, systems=["ERP"]),
            make_activity("ACT-002", "Espera de aprobación", duration=2880, systems=[]),
            make_activity("ACT-003", "Búsqueda de proveedores", duration=45, systems=["ERP"]),
            make_activity("ACT-004", "Validación de datos", duration=20, systems=["ERP"]),
        ],
        total_duration_min=2975.0,
        raw_input="texto original",
    )


SAMPLE_LLM_RESPONSE = {
    "activity_details": [
        {
            "activity_id": "ACT-001",
            "activity_name": "Revisión de solicitud",
            "waste_classification": "genera_valor",
            "waste_type": None,
            "waste_justification": "Transforma la solicitud en datos estructurados",
            "estimated_waste_time_min": 0.0,
            "is_automatable": False,
            "automation_tool": None,
            "automation_justification": None,
        },
        {
            "activity_id": "ACT-002",
            "activity_name": "Espera de aprobación",
            "waste_classification": "desperdicio",
            "waste_type": "espera",
            "waste_justification": "Genera hasta 2 días de espera sin agregar valor",
            "estimated_waste_time_min": 2880.0,
            "is_automatable": True,
            "automation_tool": "Sistema de aprobación digital (Power Automate)",
            "automation_justification": "Flujo de aprobación automático por correo",
        },
        {
            "activity_id": "ACT-003",
            "activity_name": "Búsqueda de proveedores",
            "waste_classification": "genera_valor",
            "waste_type": None,
            "waste_justification": "Identifica opciones de proveedor directamente",
            "estimated_waste_time_min": 0.0,
            "is_automatable": True,
            "automation_tool": "RPA + ERP API",
            "automation_justification": "Búsqueda automática en catálogo ERP",
        },
        {
            "activity_id": "ACT-004",
            "activity_name": "Validación de datos",
            "waste_classification": "desperdicio",
            "waste_type": "sobreproceso",
            "waste_justification": "Duplica validaciones ya realizadas en ACT-001",
            "estimated_waste_time_min": 20.0,
            "is_automatable": True,
            "automation_tool": "Validación automática ERP",
            "automation_justification": "Reglas de validación automáticas en ERP",
        },
    ],
    "redundancies": [
        {
            "activity_ids": ["ACT-001", "ACT-004"],
            "activity_names": ["Revisión de solicitud", "Validación de datos"],
            "redundancy_type": "revision_multiple",
            "description": "Ambas actividades validan los mismos datos de la solicitud",
            "suggested_action": "Combinar en una única actividad de revisión y validación",
        }
    ],
    "lean_summary": "El proceso tiene 50% de actividades de desperdicio con 2880 min de espera.",
    "six_sigma_insights": "Alta variabilidad en tiempos de espera de aprobación.",
    "kaizen_quick_wins": [
        "Implementar aprobaciones digitales para eliminar esperas",
        "Combinar revisión y validación en una sola actividad",
    ],
}


# ── Tests unitarios ───────────────────────────────────────────────────────────

def test_build_waste_analysis_totals():
    result = _build_waste_analysis(
        data=SAMPLE_LLM_RESPONSE,
        process_id="PROC-001",
        process_name="Proceso de Compras",
    )
    assert isinstance(result, WasteAnalysisResult)
    assert result.total_activities == 4
    assert result.value_added_count == 2
    assert result.waste_count == 2
    assert result.waste_percentage == 50.0
    assert result.total_waste_time_min == 2900.0


def test_build_waste_analysis_automation():
    result = _build_waste_analysis(
        data=SAMPLE_LLM_RESPONSE,
        process_id="PROC-001",
        process_name="Proceso de Compras",
    )
    assert result.automatable_count == 3
    assert result.automation_coverage_pct == 75.0


def test_build_waste_analysis_redundancies():
    result = _build_waste_analysis(
        data=SAMPLE_LLM_RESPONSE,
        process_id="PROC-001",
        process_name="Proceso de Compras",
    )
    assert len(result.redundancies) == 1
    assert "ACT-001" in result.redundancies[0].activity_ids


def test_build_waste_analysis_main_waste_types():
    result = _build_waste_analysis(
        data=SAMPLE_LLM_RESPONSE,
        process_id="PROC-001",
        process_name="Proceso de Compras",
    )
    # espera y sobreproceso deben estar en los principales
    waste_type_names = [wt.value for wt in result.main_waste_types]
    assert "espera" in waste_type_names
    assert "sobreproceso" in waste_type_names


def test_detect_sequential_waits():
    asis_json = {
        "activities": [
            {"name": "Espera de aprobación del jefe", "description": ""},
            {"name": "Reunión de equipo", "description": ""},
            {"name": "Validación de documentos pendientes", "description": ""},
        ]
    }
    insights = _detect_sequential_waits(asis_json)
    # Debe detectar al menos 2 actividades con keywords de espera/aprobación
    assert len(insights) >= 2


def test_detect_duplicate_systems():
    asis_json = {
        "activities": [
            {"name": "A1", "systems_used": ["ERP"]},
            {"name": "A2", "systems_used": ["ERP"]},
            {"name": "A3", "systems_used": ["ERP"]},
        ]
    }
    insights = _detect_duplicate_systems(asis_json)
    assert len(insights) == 1
    assert "ERP" in insights[0]


def test_detect_duplicate_systems_no_duplicates():
    asis_json = {
        "activities": [
            {"name": "A1", "systems_used": ["SAP"]},
            {"name": "A2", "systems_used": ["ERP"]},
        ]
    }
    insights = _detect_duplicate_systems(asis_json)
    assert len(insights) == 0


@patch("agent.analyzer._call_llm_with_retry")
def test_node_analyze_waste_success(mock_llm):
    mock_llm.return_value = SAMPLE_LLM_RESPONSE
    state = AgentState(
        raw_input="...",
        asis_process=make_process(),
        extraction_ok=True,
    )
    result = node_analyze_waste(state)

    assert result["analysis_ok"] is True
    assert result["waste_analysis"] is not None
    assert result["waste_analysis"].waste_count == 2


@patch("agent.analyzer._call_llm_with_retry")
def test_node_analyze_waste_llm_error(mock_llm):
    mock_llm.side_effect = Exception("Timeout de API")
    state = AgentState(
        raw_input="...",
        asis_process=make_process(),
        extraction_ok=True,
    )
    result = node_analyze_waste(state)

    assert result["analysis_ok"] is False
    assert "analyze_waste" in result["errors"][0]


def test_node_analyze_waste_no_process():
    state = AgentState(raw_input="...", asis_process=None)
    result = node_analyze_waste(state)
    assert result["analysis_ok"] is False
    assert "None" in result["errors"][0]


# ── Test de integración ───────────────────────────────────────────────────────

@pytest.mark.integration
def test_full_analysis_integration():
    """
    Test de integración real contra GPT-4o.
    Ejecutar con: pytest -m integration
    """
    state = AgentState(
        raw_input="...",
        asis_process=make_process(),
        extraction_ok=True,
    )
    result = node_analyze_waste(state)

    assert result["analysis_ok"] is True
    analysis = result["waste_analysis"]
    assert analysis.total_activities == 4
    assert analysis.waste_count >= 1
    assert len(analysis.kaizen_quick_wins) >= 1