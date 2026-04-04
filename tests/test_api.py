import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from api.main import app, _sessions
from models.schemas import AgentState, KPIReportV2, TOBEProcess


# ── Cliente de test ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_sessions():
    """Limpia el store de sesiones entre tests."""
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def completed_session(client, sample_process, full_tobe, full_analysis):
    """Crea una sesión ya completada para tests de resultados."""
    from tests.test_kpi_calculator import full_tobe, full_analysis
    from tests.test_optimizer import SAMPLE_TOBE_RESPONSE
    from tests.test_analyzer import SAMPLE_LLM_RESPONSE
    from agent.analyzer import _build_waste_analysis
    from agent.optimizer import _build_tobe_process
    from agent.bpmn_generator import build_bpmn_structure_from_tobe
    from models.schemas import BPMNOutput
    from datetime import datetime

    sid   = "test-session-completed"
    tobe  = _build_tobe_process(SAMPLE_TOBE_RESPONSE, "PROC-001")
    analysis = _build_waste_analysis(SAMPLE_LLM_RESPONSE, "PROC-001", "Test")

    state = AgentState(
        raw_input="proceso de prueba",
        asis_process=sample_process,
        extraction_ok=True,
        waste_analysis=analysis,
        analysis_ok=True,
        tobe_process=tobe,
        optimization_ok=True,
        hitl_approved=True,
        bpmn_output=BPMNOutput(
            process_id="TOBE-001",
            process_name="Test",
            xml_content="<?xml version='1.0'?><definitions/>",
            file_path="/tmp/test.bpmn",
            element_count=5,
        ),
        bpmn_ok=True,
        kpi_ok=True,
        current_node="calculate_kpis",
    )
    _sessions[sid] = state
    return sid


# ── Tests de health ───────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model" in data


# ── Tests de análisis de texto ────────────────────────────────────────────────

@patch("api.main._run_pipeline", new_callable=AsyncMock)
def test_analyze_text_success(mock_run, client):
    payload = {
        "raw_input": (
            "El proceso de contratación inicia cuando RRHH recibe una vacante. "
            "Un reclutador revisa el perfil (30 min), publica en LinkedIn (15 min), "
            "espera candidatos hasta 5 días hábiles y agenda entrevistas (60 min)."
        )
    }
    resp = client.post("/analyze/text", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    assert "session_id" in data
    assert data["status"] == "running"
    mock_run.assert_called_once()


def test_analyze_text_too_short(client):
    resp = client.post("/analyze/text", json={"raw_input": "muy corto"})
    assert resp.status_code == 422


# ── Tests de sesiones ─────────────────────────────────────────────────────────

def test_get_session_status(client, completed_session):
    resp = client.get(f"/sessions/{completed_session}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kpi_ok"] is True
    assert data["bpmn_ok"] is True
    assert data["extraction_ok"] is True


def test_get_session_not_found(client):
    resp = client.get("/sessions/inexistente-123/status")
    assert resp.status_code == 404


def test_list_sessions(client, completed_session):
    resp = client.get("/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1


def test_delete_session(client, completed_session):
    resp = client.delete(f"/sessions/{completed_session}")
    assert resp.status_code == 200
    assert completed_session not in _sessions


# ── Tests de resultados ───────────────────────────────────────────────────────

def test_get_asis_process(client, completed_session):
    resp = client.get(f"/sessions/{completed_session}/process")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "activities" in data


def test_get_waste_analysis(client, completed_session):
    resp = client.get(f"/sessions/{completed_session}/analysis")
    assert resp.status_code == 200
    data = resp.json()
    assert "waste_percentage" in data
    assert "activity_details" in data


def test_get_tobe_process(client, completed_session):
    resp = client.get(f"/sessions/{completed_session}/tobe")
    assert resp.status_code == 200
    data = resp.json()
    assert "activities" in data
    assert "sipoc" in data


def test_get_full_report(client, completed_session):
    resp = client.get(f"/sessions/{completed_session}/report")
    assert resp.status_code == 200
    data = resp.json()
    assert all(k in data for k in [
        "asis_process", "waste_analysis", "tobe_process", "kpi_report"
    ])


def test_get_result_not_ready(client):
    # Sesión sin análisis completado
    _sessions["pending"] = AgentState(
        raw_input="...",
        current_node="extract_asis",
    )
    resp = client.get("/sessions/pending/process")
    assert resp.status_code == 425   # Too Early


# ── Tests HITL ────────────────────────────────────────────────────────────────

@patch("api.main._resume_pipeline", new_callable=AsyncMock)
def test_hitl_approve(mock_resume, client, sample_process):
    from tests.test_optimizer import SAMPLE_TOBE_RESPONSE
    from agent.optimizer import _build_tobe_process

    sid   = "hitl-session"
    tobe  = _build_tobe_process(SAMPLE_TOBE_RESPONSE, "PROC-001")
    state = AgentState(
        raw_input="...",
        asis_process=sample_process,
        tobe_process=tobe,
        optimization_ok=True,
        hitl_required=True,
        current_node="hitl_review",
    )
    _sessions[sid] = state

    resp = client.post(
        f"/sessions/{sid}/review",
        json={"approved": True, "feedback": "Se ve bien, adelante."},
    )
    assert resp.status_code == 200
    assert _sessions[sid].hitl_approved is True
    assert _sessions[sid].tobe_process.human_approved is True
    mock_resume.assert_called_once()


@patch("api.main._resume_pipeline", new_callable=AsyncMock)
def test_hitl_reject_requires_feedback(mock_resume, client, sample_process):
    from tests.test_optimizer import SAMPLE_TOBE_RESPONSE
    from agent.optimizer import _build_tobe_process

    sid   = "hitl-reject"
    tobe  = _build_tobe_process(SAMPLE_TOBE_RESPONSE, "PROC-001")
    _sessions[sid] = AgentState(
        raw_input="...",
        asis_process=sample_process,
        tobe_process=tobe,
        hitl_required=True,
    )

    # Sin feedback → debe fallar con 422
    resp = client.post(
        f"/sessions/{sid}/review",
        json={"approved": False, "feedback": ""},
    )
    assert resp.status_code == 422