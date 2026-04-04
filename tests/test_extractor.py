import pytest
from unittest.mock import patch, MagicMock
from models.schemas import AgentState, Process
from agent.process_extractor import node_extract_asis, _build_process_from_dict


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_ASIS = """
El proceso de contratación de proveedores inicia cuando el área de compras
recibe una solicitud de adquisición. Un analista revisa los requisitos (30 min),
busca proveedores en el sistema ERP (45 min), solicita cotizaciones por correo (20 min),
espera respuestas hasta 3 días hábiles, compara ofertas en Excel (60 min),
elabora un informe de recomendación (90 min) y lo envía al jefe de compras
para aprobación final (15 min). Si es aprobado, se genera la orden de compra en SAP.
"""

SAMPLE_LLM_RESPONSE = {
    "id": "PROC-001",
    "name": "Contratación de Proveedores",
    "description": "Proceso de selección y contratación de proveedores",
    "owner": "Área de Compras",
    "scope": "Desde solicitud de adquisición hasta generación de orden de compra",
    "participants": ["Analista de Compras", "Jefe de Compras"],
    "systems": ["ERP", "SAP", "Excel", "Correo electrónico"],
    "total_duration_min": 260.0,
    "activities": [
        {
            "id": "ACT-001",
            "name": "Revisión de requisitos",
            "description": "El analista revisa los requisitos de la solicitud",
            "responsible": "Analista de Compras",
            "type": "analitica",
            "estimated_duration_min": 30.0,
            "depends_on": [],
            "systems_used": [],
            "subactivities": [],
        },
        {
            "id": "ACT-002",
            "name": "Búsqueda de proveedores",
            "description": "Búsqueda en el sistema ERP",
            "responsible": "Analista de Compras",
            "type": "operativa",
            "estimated_duration_min": 45.0,
            "depends_on": ["ACT-001"],
            "systems_used": ["ERP"],
            "subactivities": [],
        },
    ],
}


# ── Tests unitarios ───────────────────────────────────────────────────────────

def test_build_process_from_dict():
    process = _build_process_from_dict(SAMPLE_LLM_RESPONSE, SAMPLE_ASIS)
    assert isinstance(process, Process)
    assert process.name == "Contratación de Proveedores"
    assert len(process.activities) == 2
    assert process.activities[0].id == "ACT-001"
    assert process.total_duration_min == 75.0  # suma real de las 2 actividades


def test_build_process_calculates_duration_when_missing():
    data = {**SAMPLE_LLM_RESPONSE, "total_duration_min": None}
    process = _build_process_from_dict(data, SAMPLE_ASIS)
    # Debe calcularlo sumando las actividades
    assert process.total_duration_min == 75.0


@patch("agent.process_extractor._call_llm_with_retry")
def test_node_extract_asis_success(mock_llm):
    mock_llm.return_value = SAMPLE_LLM_RESPONSE
    state = AgentState(raw_input=SAMPLE_ASIS)

    result = node_extract_asis(state)

    assert result["extraction_ok"] is True
    assert result["asis_process"] is not None
    assert result["asis_process"].name == "Contratación de Proveedores"
    assert len(result["asis_process"].activities) == 2


@patch("agent.process_extractor._call_llm_with_retry")
def test_node_extract_asis_llm_returns_invalid(mock_llm):
    # LLM retorna dict sin campos mínimos
    mock_llm.side_effect = ValueError("campos mínimos faltantes")
    state = AgentState(raw_input=SAMPLE_ASIS)

    result = node_extract_asis(state)

    assert result["extraction_ok"] is False
    assert len(result["errors"]) > 0
    assert "extract_asis" in result["errors"][0]


def test_node_extract_asis_empty_input():
    state = AgentState(raw_input="   ")
    result = node_extract_asis(state)
    assert result["extraction_ok"] is False
    assert "vacío" in result["errors"][0]


# ── Tests de integración (requieren OPENAI_API_KEY real) ─────────────────────

@pytest.mark.integration
def test_full_extraction_integration():
    """
    Test de integración real contra GPT-4o.
    Ejecutar con: pytest -m integration
    """
    state = AgentState(raw_input=SAMPLE_ASIS)
    result = node_extract_asis(state)

    assert result["extraction_ok"] is True
    process = result["asis_process"]
    assert len(process.activities) >= 4
    assert process.total_duration_min > 0