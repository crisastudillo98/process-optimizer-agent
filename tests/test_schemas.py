"""
Tests de validación de todos los schemas Pydantic v2.
Verifica contratos de datos entre nodos del grafo.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError
from models.schemas import (
    Activity, ActivityType, SubActivity,
    Process, WasteClassification, WasteType,
    ActivityWasteDetail, Redundancy, WasteAnalysisResult,
    OptimizedActivity, ActivityStatus, TOBEProcess,
    EnrichedKPI, KPIEnrichment, KPIReportV2,
    BPMNElement, BPMNElementType, BPMNSequenceFlow,
    BPMNStructure, BPMNOutput,
    AgentState,
)


# ── Activity ──────────────────────────────────────────────────────────────────

def test_activity_valid():
    act = Activity(
        id="ACT-001",
        name="Revisión de datos",
        description="Revisa los datos del formulario",
        responsible="Analista",
        type=ActivityType.ANALYTICAL,
        estimated_duration_min=30.0,
    )
    assert act.id == "ACT-001"
    assert act.depends_on == []
    assert act.subactivities == []


def test_activity_invalid_missing_required():
    with pytest.raises(ValidationError):
        Activity(name="Sin ID")   # falta id, description, responsible, type, duration


def test_activity_type_enum_values():
    types = [t.value for t in ActivityType]
    assert "operativa" in types
    assert "analitica" in types
    assert "cognitiva" in types


# ── Process ───────────────────────────────────────────────────────────────────

def test_process_valid(sample_process):
    assert sample_process.id == "PROC-TEST"
    assert len(sample_process.activities) == 3
    assert sample_process.total_duration_min > 0


def test_process_empty_activities():
    p = Process(
        id="PROC-EMPTY",
        name="Vacío",
        description="Sin actividades",
        owner="Test",
        scope="N/A",
    )
    assert p.activities == []
    assert p.total_duration_min == 0.0


# ── WasteAnalysisResult ───────────────────────────────────────────────────────

def test_waste_analysis_result_percentages():
    detail = ActivityWasteDetail(
        activity_id="ACT-001",
        activity_name="Test",
        waste_classification=WasteClassification.WASTE,
        waste_type=WasteType.WAITING,
        waste_justification="Genera espera innecesaria",
        estimated_waste_time_min=120.0,
    )
    result = WasteAnalysisResult(
        process_id="PROC-001",
        process_name="Test",
        activity_details=[detail],
        waste_count=1,
        total_activities=1,
        waste_percentage=100.0,
        lean_summary="Todo es desperdicio",
        six_sigma_insights="Alta variabilidad",
    )
    assert result.waste_percentage == 100.0
    assert result.total_waste_time_min == 0.0    # no se suma automáticamente aquí


def test_waste_type_all_8_mudas():
    mudas = [wt.value for wt in WasteType]
    expected = [
        "espera", "sobreproceso", "defectos",
        "sobreproduccion", "transporte",
        "inventario", "movimiento", "talento_no_utilizado"
    ]
    for muda in expected:
        assert muda in mudas


# ── TOBEProcess ───────────────────────────────────────────────────────────────

def test_tobe_process_default_methodologies():
    tobe = TOBEProcess(
        id="TOBE-001",
        original_process_id="PROC-001",
        name="TO-BE Test",
        description="Test",
        owner="Test",
        total_duration_min=100.0,
    )
    assert "Lean" in tobe.applied_methodologies
    assert "Six Sigma" in tobe.applied_methodologies
    assert "Kaizen" in tobe.applied_methodologies
    assert tobe.human_approved is False


def test_optimized_activity_eliminated_status():
    act = OptimizedActivity(
        id="OPT-001",
        name="Actividad eliminada",
        description="Esta se elimina",
        responsible="Nadie",
        type=ActivityType.OPERATIVE,
        status=ActivityStatus.ELIMINATED,
        estimated_duration_min=0.0,
        duration_reduction_pct=100.0,
        improvement_justification="Muda pura",
    )
    assert act.status == ActivityStatus.ELIMINATED
    assert act.duration_reduction_pct == 100.0


# ── EnrichedKPI ───────────────────────────────────────────────────────────────

def test_enriched_kpi_without_enrichment():
    kpi = EnrichedKPI(
        name="Test KPI",
        unit="min",
        asis_value=100.0,
        tobe_value=40.0,
        reduction_absolute=60.0,
        reduction_pct=60.0,
        interpretation="Reducción del 60%",
    )
    assert kpi.enrichment is None
    assert kpi.reduction_pct == 60.0


def test_enriched_kpi_with_enrichment():
    enrichment = KPIEnrichment(
        business_interpretation="Ahorra 1h por proceso",
        industry_benchmark="Benchmark Lean: >50% reducción",
        implementation_risk="Resistencia al cambio del equipo",
        next_step="Implementar aprobaciones digitales en semana 1",
    )
    kpi = EnrichedKPI(
        name="Ciclo",
        unit="min",
        asis_value=100.0,
        tobe_value=40.0,
        reduction_absolute=60.0,
        reduction_pct=60.0,
        interpretation="Reducción del 60%",
        enrichment=enrichment,
    )
    assert kpi.enrichment.next_step != ""


# ── BPMNStructure ─────────────────────────────────────────────────────────────

def test_bpmn_element_types():
    types = [t.value for t in BPMNElementType]
    assert "startEvent" in types
    assert "endEvent" in types
    assert "serviceTask" in types
    assert "exclusiveGateway" in types


def test_bpmn_sequence_flow_optional_name():
    flow = BPMNSequenceFlow(
        id="flow_1",
        source_ref="start_1",
        target_ref="task_1",
    )
    assert flow.name is None


def test_bpmn_structure_empty():
    s = BPMNStructure(
        process_id="P1",
        process_name="Test",
    )
    assert s.elements == []
    assert s.sequence_flows == []
    assert s.lanes == []


# ── AgentState ────────────────────────────────────────────────────────────────

def test_agent_state_defaults():
    state = AgentState()
    assert state.extraction_ok is False
    assert state.analysis_ok is False
    assert state.optimization_ok is False
    assert state.hitl_required is False
    assert state.hitl_approved is False
    assert state.bpmn_ok is False
    assert state.kpi_ok is False
    assert state.errors == []
    assert state.rag_context == []
    assert state.current_node == "start"


def test_agent_state_serialization(sample_agent_state):
    dumped = sample_agent_state.model_dump()
    restored = AgentState(**dumped)
    assert restored.extraction_ok == sample_agent_state.extraction_ok
    assert restored.asis_process.id == sample_agent_state.asis_process.id


def test_agent_state_error_accumulation():
    state = AgentState()
    state.errors.append("Error 1")
    state.errors.append("Error 2")
    assert len(state.errors) == 2
    assert "Error 1" in state.errors