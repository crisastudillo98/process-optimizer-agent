import pytest
from unittest.mock import patch, MagicMock
from models.schemas import (
    AgentState, ActivityStatus, ActivityType,
    EnrichedKPI, KPIReportV2,
)
from agent.kpi_calculator import (
    calculate_kpis,
    node_calculate_kpis,
    _calc_cycle_time,
    _calc_headcount,
    _calc_waste_reduction,
    _calc_automation_coverage,
    _calc_process_efficiency,
    _calc_roi,
    _estimate_sigma_level,
)
from tests.test_analyzer import SAMPLE_LLM_RESPONSE
from tests.test_optimizer import SAMPLE_TOBE_RESPONSE
from agent.analyzer import _build_waste_analysis
from agent.optimizer import _build_tobe_process


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def full_tobe(sample_process):
    return _build_tobe_process(SAMPLE_TOBE_RESPONSE, sample_process.id)


@pytest.fixture
def full_analysis(sample_process):
    return _build_waste_analysis(
        SAMPLE_LLM_RESPONSE, sample_process.id, sample_process.name
    )


# ── Tests KPIs individuales ───────────────────────────────────────────────────

def test_calc_cycle_time_reduction(sample_process, full_tobe):
    kpi = _calc_cycle_time(sample_process, full_tobe)
    assert isinstance(kpi, EnrichedKPI)
    assert kpi.asis_value == sample_process.total_duration_min
    assert kpi.tobe_value == full_tobe.total_duration_min
    assert kpi.reduction_pct > 0
    assert kpi.reduction_absolute > 0


def test_calc_cycle_time_zero_duration(sample_process, full_tobe):
    sample_process.total_duration_min = 0.0
    kpi = _calc_cycle_time(sample_process, full_tobe)
    assert kpi.reduction_pct == 0.0


def test_calc_headcount_reduction(sample_process, full_tobe):
    kpi = _calc_headcount(sample_process, full_tobe)
    assert isinstance(kpi, EnrichedKPI)
    assert kpi.asis_value == float(len(sample_process.activities))
    assert kpi.tobe_value >= 0.0


def test_calc_waste_reduction(full_analysis, full_tobe):
    kpi = _calc_waste_reduction(full_analysis, full_tobe)
    assert isinstance(kpi, EnrichedKPI)
    assert kpi.asis_value == full_analysis.total_waste_time_min
    assert kpi.tobe_value < kpi.asis_value
    assert kpi.reduction_pct > 0


def test_calc_automation_coverage(sample_process, full_tobe, full_analysis):
    kpi = _calc_automation_coverage(sample_process, full_tobe, full_analysis)
    assert isinstance(kpi, EnrichedKPI)
    assert 0.0 <= kpi.tobe_value <= 100.0


def test_calc_process_efficiency(sample_process, full_tobe, full_analysis):
    kpi = _calc_process_efficiency(sample_process, full_tobe, full_analysis)
    assert isinstance(kpi, EnrichedKPI)
    assert 0.0 <= kpi.asis_value <= 100.0
    assert 0.0 <= kpi.tobe_value <= 100.0


# ── Tests ROI ─────────────────────────────────────────────────────────────────

def test_calc_roi_positive(sample_process, full_tobe):
    cycle_kpi    = _calc_cycle_time(sample_process, full_tobe)
    headcount_kpi = _calc_headcount(sample_process, full_tobe)
    roi, payback, annual_hrs = _calc_roi(
        sample_process, full_tobe, cycle_kpi, headcount_kpi,
        cost_per_hour_usd=25.0,
        implementation_cost_usd=1000.0,
    )
    assert isinstance(roi, float)
    assert isinstance(payback, float)
    assert annual_hrs >= 0.0


def test_calc_roi_zero_implementation_cost(sample_process, full_tobe):
    cycle_kpi    = _calc_cycle_time(sample_process, full_tobe)
    headcount_kpi = _calc_headcount(sample_process, full_tobe)
    roi, _, _ = _calc_roi(
        sample_process, full_tobe, cycle_kpi, headcount_kpi,
        implementation_cost_usd=0.0,
    )
    assert roi == 0.0


# ── Tests nivel Sigma ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("waste_pct,expected_sigma", [
    (60.0, 2.0),
    (40.0, 3.0),
    (20.0, 4.0),
    (10.0, 5.0),
    (3.0,  6.0),
])
def test_estimate_sigma_level(waste_pct, expected_sigma):
    assert _estimate_sigma_level(waste_pct) == expected_sigma


# ── Tests del reporte completo ────────────────────────────────────────────────

@patch("agent.kpi_calculator._enrich_kpis_with_llm")
def test_calculate_kpis_full_report(mock_enrich, sample_process, full_tobe, full_analysis):
    # Simula enriquecimiento LLM vacío
    mock_enrich.return_value = {
        "executive_summary": "Resumen ejecutivo simulado.",
        "kpi_enrichments": {},
    }

    report = calculate_kpis(
        process=sample_process,
        tobe=full_tobe,
        analysis=full_analysis,
        enrich_with_llm=True,
    )

    assert isinstance(report, KPIReportV2)
    assert report.cycle_time.reduction_pct > 0
    assert report.estimated_roi_pct is not None
    assert report.sigma_level_asis is not None
    assert report.sigma_level_tobe is not None
    assert report.executive_summary != ""


def test_calculate_kpis_without_llm(sample_process, full_tobe, full_analysis):
    report = calculate_kpis(
        process=sample_process,
        tobe=full_tobe,
        analysis=full_analysis,
        enrich_with_llm=False,    # sin LLM — puro determinístico
    )
    assert isinstance(report, KPIReportV2)
    assert report.cycle_time.enrichment is None
    assert report.executive_summary != ""


def test_calculate_kpis_llm_failure_non_blocking(
    sample_process, full_tobe, full_analysis
):
    with patch(
        "agent.kpi_calculator._enrich_kpis_with_llm",
        side_effect=Exception("API timeout"),
    ):
        # No debe lanzar excepción — el enriquecimiento falla silenciosamente
        report = calculate_kpis(
            process=sample_process,
            tobe=full_tobe,
            analysis=full_analysis,
            enrich_with_llm=True,
        )
        assert isinstance(report, KPIReportV2)
        assert report.cycle_time.enrichment is None


# ── Tests del nodo LangGraph ──────────────────────────────────────────────────

@patch("agent.kpi_calculator.calculate_kpis")
@patch("agent.kpi_calculator._persist_case_to_rag")
def test_node_calculate_kpis_success(
    mock_persist, mock_calc, sample_agent_state, full_tobe, full_analysis
):
    mock_report = MagicMock(spec=KPIReportV2)
    mock_calc.return_value = mock_report

    sample_agent_state.tobe_process   = full_tobe
    sample_agent_state.waste_analysis = full_analysis

    result = node_calculate_kpis(sample_agent_state)

    assert result["kpi_ok"] is True
    assert result["kpi_report"] == mock_report
    mock_persist.assert_called_once()


def test_node_calculate_kpis_missing_fields(sample_agent_state):
    sample_agent_state.tobe_process   = None
    sample_agent_state.waste_analysis = None

    result = node_calculate_kpis(sample_agent_state)

    assert result["kpi_ok"] is False
    assert "faltan campos" in result["errors"][-1]


@patch("agent.kpi_calculator.calculate_kpis")
def test_node_calculate_kpis_exception(mock_calc, sample_agent_state, full_tobe, full_analysis):
    mock_calc.side_effect = Exception("Error inesperado")
    sample_agent_state.tobe_process   = full_tobe
    sample_agent_state.waste_analysis = full_analysis

    result = node_calculate_kpis(sample_agent_state)

    assert result["kpi_ok"] is False
    assert "calculate_kpis" in result["errors"][-1]


# ── Test de integración ───────────────────────────────────────────────────────

@pytest.mark.integration
def test_full_kpi_integration(sample_process, full_tobe, full_analysis):
    """
    Test de integración real contra GPT-4o para el enriquecimiento.
    Ejecutar con: pytest -m integration
    """
    report = calculate_kpis(
        process=sample_process,
        tobe=full_tobe,
        analysis=full_analysis,
        enrich_with_llm=True,
    )

    assert report.cycle_time.enrichment is not None
    assert report.cycle_time.enrichment.business_interpretation != ""
    assert report.cycle_time.enrichment.next_step != ""
    assert report.executive_summary != ""
    assert report.estimated_roi_pct is not None