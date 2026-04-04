import pytest
from pathlib import Path
from models.schemas import (
    TOBEProcess, OptimizedActivity,
    ActivityStatus, ActivityType,
    AgentState, BPMNStructure, BPMNOutput,
)
from agent.bpmn_generator import (
    build_bpmn_structure_from_tobe,
    _build_xml_from_structure,
    _validate_bpmn_xml,
    _save_bpmn_file,
    _map_activity_to_bpmn_type,
    _needs_gateway,
    node_generate_bpmn,
    BPMNElementType,
)
from lxml import etree


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_optimized_activity(
    id: str,
    name: str,
    status: ActivityStatus = ActivityStatus.KEPT,
    act_type: ActivityType = ActivityType.OPERATIVE,
    duration: float = 30.0,
    automation_tool: str | None = None,
) -> OptimizedActivity:
    return OptimizedActivity(
        id=id,
        original_activity_id=f"ACT-{id[-3:]}",
        name=name,
        description=f"Descripción de {name}",
        responsible="Analista",
        type=act_type,
        status=status,
        estimated_duration_min=duration,
        duration_reduction_pct=0.0,
        is_automatable=automation_tool is not None,
        automation_tool=automation_tool,
        improvement_justification="Mejora Lean aplicada",
    )


def make_tobe_process() -> TOBEProcess:
    return TOBEProcess(
        id="TOBE-001",
        original_process_id="PROC-001",
        name="Proceso de Compras Optimizado",
        description="TO-BE optimizado con Lean Six Sigma",
        owner="Compras",
        activities=[
            make_optimized_activity(
                "OPT-001", "Revisión y validación integrada",
                ActivityStatus.COMBINED, ActivityType.ANALYTICAL, 25.0
            ),
            make_optimized_activity(
                "OPT-002", "Búsqueda automática de proveedores",
                ActivityStatus.AUTOMATED, ActivityType.OPERATIVE, 5.0,
                "UiPath + ERP API"
            ),
            make_optimized_activity(
                "OPT-003", "Aprobación digital",
                ActivityStatus.OPTIMIZED, ActivityType.COGNITIVE, 120.0,
                "Power Automate"
            ),
            make_optimized_activity(
                "OPT-004", "Tarea eliminada",
                ActivityStatus.ELIMINATED, ActivityType.OPERATIVE, 60.0
            ),
        ],
        total_duration_min=150.0,
    )


# ── Tests de mapeo ────────────────────────────────────────────────────────────

def test_map_automated_to_service_task():
    act = make_optimized_activity("OPT-001", "Test", ActivityStatus.AUTOMATED)
    assert _map_activity_to_bpmn_type(act) == BPMNElementType.SERVICE_TASK


def test_map_cognitive_to_user_task():
    act = make_optimized_activity(
        "OPT-001", "Test", ActivityStatus.KEPT, ActivityType.COGNITIVE
    )
    assert _map_activity_to_bpmn_type(act) == BPMNElementType.USER_TASK


def test_map_operative_to_task():
    act = make_optimized_activity(
        "OPT-001", "Test", ActivityStatus.KEPT, ActivityType.OPERATIVE
    )
    assert _map_activity_to_bpmn_type(act) == BPMNElementType.TASK


def test_needs_gateway_true():
    assert _needs_gateway("Aprobación del jefe de área") is True
    assert _needs_gateway("Validación de documentos") is True
    assert _needs_gateway("Revisión de solicitud") is True


def test_needs_gateway_false():
    assert _needs_gateway("Registro en sistema") is False
    assert _needs_gateway("Envío de correo") is False


# ── Tests de estructura ───────────────────────────────────────────────────────

def test_build_structure_excludes_eliminated():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)

    element_names = [e.name for e in structure.elements]
    assert "Tarea eliminada" not in element_names


def test_build_structure_has_start_and_end():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)

    types = [e.type for e in structure.elements]
    assert BPMNElementType.START_EVENT in types
    assert BPMNElementType.END_EVENT in types


def test_build_structure_service_task_for_automated():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)

    service_tasks = [
        e for e in structure.elements
        if e.type == BPMNElementType.SERVICE_TASK
    ]
    # OPT-002 y OPT-003 son automatizadas o tienen automation_tool
    assert len(service_tasks) >= 1


def test_build_structure_lanes_by_responsible():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)
    # Todos tienen responsible="Analista" → 1 lane
    assert len(structure.lanes) == 1
    assert structure.lanes[0].name == "Analista"


def test_build_structure_sequence_flows_connected():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)
    # Debe haber al menos n+1 flujos para n actividades + start + end
    assert len(structure.sequence_flows) >= len(structure.elements) - 1


# ── Tests de XML ─────────────────────────────────────────────────────────────

def test_build_xml_is_valid_xml():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)
    xml = _build_xml_from_structure(structure)

    # Debe parsear sin errores
    root = etree.fromstring(xml.encode("utf-8"))
    assert root is not None


def test_build_xml_has_bpmn_namespace():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)
    xml = _build_xml_from_structure(structure)

    assert "http://www.omg.org/spec/BPMN/20100524/MODEL" in xml


def test_build_xml_has_process_element():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)
    xml = _build_xml_from_structure(structure)

    root = etree.fromstring(xml.encode("utf-8"))
    ns   = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
    processes = root.findall(".//bpmn:process", ns)
    assert len(processes) == 1


def test_validate_bpmn_valid_xml():
    tobe = make_tobe_process()
    structure = build_bpmn_structure_from_tobe(tobe)
    xml    = _build_xml_from_structure(structure)
    errors = _validate_bpmn_xml(xml)
    assert errors == []


def test_validate_bpmn_invalid_xml():
    errors = _validate_bpmn_xml("<invalid>xml without bpmn</invalid>")
    assert len(errors) > 0


def test_validate_bpmn_missing_start_event():
    # XML mínimo sin startEvent
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
        <process id="p1"><endEvent id="end1"/></process>
    </definitions>"""
    errors = _validate_bpmn_xml(xml)
    assert any("startEvent" in e for e in errors)


# ── Tests de persistencia ────────────────────────────────────────────────────

def test_save_bpmn_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.bpmn_generator.settings",
        type("S", (), {"bpmn_output_path": str(tmp_path)})(),
    )
    xml_content = '<?xml version="1.0"?><root/>'
    file_path   = _save_bpmn_file(xml_content, "Proceso de Prueba")

    assert Path(file_path).exists()
    assert Path(file_path).suffix == ".bpmn"
    assert "proceso_de_prueba" in Path(file_path).name


# ── Tests del nodo LangGraph ──────────────────────────────────────────────────

def test_node_generate_bpmn_success(monkeypatch):
    monkeypatch.setattr("agent.bpmn_generator.settings", type("S", (), {
        "hitl_enabled":    False,
        "bpmn_output_path": "/tmp/bpmn_test",
    })())

    tobe  = make_tobe_process()
    state = AgentState(
        raw_input="...",
        tobe_process=tobe,
        hitl_approved=True,
        optimization_ok=True,
    )

    result = node_generate_bpmn(state)

    assert result["bpmn_ok"] is True
    assert result["bpmn_output"] is not None
    assert result["bpmn_output"].element_count > 0


def test_node_generate_bpmn_no_tobe():
    state = AgentState(raw_input="...", tobe_process=None)
    result = node_generate_bpmn(state)
    assert result["bpmn_ok"] is False
    assert "None" in result["errors"][-1]


def test_node_generate_bpmn_hitl_not_approved(monkeypatch):
    monkeypatch.setattr("agent.bpmn_generator.settings", type("S", (), {
        "hitl_enabled": True,
        "bpmn_output_path": "/tmp/bpmn_test",
    })())

    state = AgentState(
        raw_input="...",
        tobe_process=make_tobe_process(),
        hitl_approved=False,
    )
    result = node_generate_bpmn(state)
    assert result["bpmn_ok"] is False
    assert "aprobado" in result["errors"][-1]