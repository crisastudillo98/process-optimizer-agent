import pytest
from models.schemas import (
    AgentState, Process, Activity, ActivityType,
    WasteAnalysisResult, ActivityWasteDetail,
    WasteClassification, WasteType,
)
from datetime import datetime


@pytest.fixture
def sample_activity_factory():
    def _make(id: str, name: str, duration: float = 30.0,
               systems: list = None, type: ActivityType = ActivityType.OPERATIVE):
        return Activity(
            id=id, name=name,
            description=f"Descripción de {name}",
            responsible="Analista",
            type=type,
            estimated_duration_min=duration,
            depends_on=[],
            systems_used=systems or [],
        )
    return _make


@pytest.fixture
def sample_process(sample_activity_factory) -> Process:
    return Process(
        id="PROC-TEST",
        name="Proceso de Prueba",
        description="Proceso de prueba para tests",
        owner="QA Team",
        scope="Inicio → Fin",
        participants=["Analista", "Jefe"],
        systems=["ERP"],
        activities=[
            sample_activity_factory("ACT-001", "Actividad de valor", 30, ["ERP"]),
            sample_activity_factory("ACT-002", "Espera de aprobación", 1440, []),
            sample_activity_factory("ACT-003", "Registro en sistema", 20, ["ERP"]),
        ],
        total_duration_min=1490.0,
        raw_input="Proceso de prueba",
    )


@pytest.fixture
def sample_agent_state(sample_process) -> AgentState:
    return AgentState(
        raw_input="Proceso de prueba",
        asis_process=sample_process,
        extraction_ok=True,
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests de integración real contra APIs externas"
    )