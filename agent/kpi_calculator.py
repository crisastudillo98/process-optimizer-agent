from __future__ import annotations
import json
from tenacity import retry, stop_after_attempt, wait_exponential

#from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
from models.schemas import (
    AgentState,
    Process,
    TOBEProcess,
    WasteAnalysisResult,
    ActivityStatus,
    EnrichedKPI,
    KPIEnrichment,
    KPIReportV2,
)
from prompts.kpi_estimation import kpi_estimation_prompt
from observability.logger import get_logger
from llm.factory import get_llm

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CÁLCULOS DETERMINÍSTICOS (sin LLM)
# ─────────────────────────────────────────────

def _calc_cycle_time(
    asis: Process,
    tobe: TOBEProcess,
) -> EnrichedKPI:
    """
    KPI 1 — Reducción de tiempo de ciclo.
    Compara duración total AS-IS vs TO-BE en minutos y horas.
    """
    asis_val = asis.total_duration_min
    tobe_val = tobe.total_duration_min
    reduction_abs = asis_val - tobe_val
    reduction_pct = round((reduction_abs / asis_val * 100), 1) if asis_val > 0 else 0.0

    return EnrichedKPI(
        name="Reducción de Tiempo de Ciclo",
        unit="minutos",
        asis_value=round(asis_val, 1),
        tobe_value=round(tobe_val, 1),
        reduction_absolute=round(reduction_abs, 1),
        reduction_pct=reduction_pct,
        interpretation=(
            f"El tiempo de ciclo se reduce de "
            f"{asis_val/60:.1f}h a {tobe_val/60:.1f}h "
            f"({reduction_pct:.1f}% de mejora)"
        ),
    )


def _calc_headcount(
    asis: Process,
    tobe: TOBEProcess,
) -> EnrichedKPI:
    """
    KPI 2 — Reducción de actividades manuales (proxy de headcount).
    Compara actividades no-automatizadas AS-IS vs TO-BE.
    """
    asis_manual = len(asis.activities)

    tobe_manual = sum(
        1 for a in tobe.activities
        if a.status != ActivityStatus.ELIMINATED
        and not a.is_automatable
    )

    reduction_abs = asis_manual - tobe_manual
    reduction_pct = round(
        (reduction_abs / asis_manual * 100), 1
    ) if asis_manual > 0 else 0.0

    return EnrichedKPI(
        name="Reducción de Actividades Manuales",
        unit="actividades",
        asis_value=float(asis_manual),
        tobe_value=float(tobe_manual),
        reduction_absolute=float(reduction_abs),
        reduction_pct=reduction_pct,
        interpretation=(
            f"Las actividades manuales se reducen de "
            f"{asis_manual} a {tobe_manual} "
            f"({reduction_pct:.1f}% liberado para tareas de mayor valor)"
        ),
    )


def _calc_waste_reduction(
    analysis: WasteAnalysisResult,
    tobe: TOBEProcess,
) -> EnrichedKPI:
    """
    KPI 3 — Reducción de desperdicios (Muda eliminada).
    Compara tiempo de desperdicio AS-IS vs tiempo residual estimado TO-BE.
    """
    asis_waste_time = analysis.total_waste_time_min

    # Tiempo residual: actividades del TO-BE con status optimizada
    # conservan ~30% del desperdicio original estimado
    eliminated_count = sum(
        1 for a in tobe.activities
        if a.status == ActivityStatus.ELIMINATED
    )
    optimized_count = sum(
        1 for a in tobe.activities
        if a.status == ActivityStatus.OPTIMIZED
    )
    automated_count = sum(
        1 for a in tobe.activities
        if a.status == ActivityStatus.AUTOMATED
    )

    # Estimación conservadora del desperdicio residual
    # eliminadas → 100% waste removida
    # automatizadas → 85% waste removida
    # optimizadas → 60% waste removida
    total_activities = len(analysis.activity_details) or 1
    weight = (
        (eliminated_count * 1.0) +
        (automated_count * 0.85) +
        (optimized_count * 0.60)
    ) / total_activities

    tobe_waste_time = round(asis_waste_time * (1 - weight), 1)
    reduction_abs   = round(asis_waste_time - tobe_waste_time, 1)
    reduction_pct   = round(
        (reduction_abs / asis_waste_time * 100), 1
    ) if asis_waste_time > 0 else 0.0

    return EnrichedKPI(
        name="Reducción de Tiempo de Desperdicio (Muda)",
        unit="minutos",
        asis_value=round(asis_waste_time, 1),
        tobe_value=tobe_waste_time,
        reduction_absolute=reduction_abs,
        reduction_pct=reduction_pct,
        interpretation=(
            f"El tiempo de desperdicio se reduce de "
            f"{asis_waste_time/60:.1f}h a {tobe_waste_time/60:.1f}h — "
            f"{reduction_pct:.1f}% de Muda eliminada"
        ),
    )


def _calc_automation_coverage(
    asis: Process,
    tobe: TOBEProcess,
    analysis: WasteAnalysisResult,
) -> EnrichedKPI:
    """
    KPI 4 — Cobertura de automatización.
    % de actividades del TO-BE que son automatizadas vs AS-IS.
    """
    asis_automated_pct = analysis.automation_coverage_pct

    tobe_active = [
        a for a in tobe.activities
        if a.status != ActivityStatus.ELIMINATED
    ]
    tobe_automated = sum(1 for a in tobe_active if a.is_automatable)
    tobe_automated_pct = round(
        (tobe_automated / len(tobe_active) * 100), 1
    ) if tobe_active else 0.0

    delta = round(tobe_automated_pct - asis_automated_pct, 1)

    return EnrichedKPI(
        name="Cobertura de Automatización",
        unit="%",
        asis_value=asis_automated_pct,
        tobe_value=tobe_automated_pct,
        reduction_absolute=delta,        # aquí es incremento
        reduction_pct=delta,
        interpretation=(
            f"La automatización sube de {asis_automated_pct:.1f}% "
            f"a {tobe_automated_pct:.1f}% del total de actividades "
            f"(+{delta:.1f} puntos porcentuales)"
        ),
    )


def _calc_process_efficiency(
    asis: Process,
    tobe: TOBEProcess,
    analysis: WasteAnalysisResult,
) -> EnrichedKPI:
    """
    KPI 5 — Eficiencia del proceso (Value-Added Ratio).
    Process Time (actividades de valor) / Lead Time total * 100.
    Métrica estándar Lean: proceso saludable > 50%.
    """
    # AS-IS: solo actividades que generan valor / total
    asis_value_time = sum(
        a.estimated_duration_min
        for a in asis.activities
        if a.waste_classification is None
        or a.waste_classification.value == "genera_valor"
    )
    asis_efficiency = round(
        (asis_value_time / asis.total_duration_min * 100), 1
    ) if asis.total_duration_min > 0 else 0.0

    # TO-BE: excluimos eliminadas del lead time
    tobe_total = tobe.total_duration_min or 1.0
    tobe_value_time = sum(
        a.estimated_duration_min
        for a in tobe.activities
        if a.status not in (ActivityStatus.ELIMINATED,)
    )
    tobe_efficiency = round(
        (tobe_value_time / tobe_total * 100), 1
    ) if tobe_total > 0 else 0.0

    delta = round(tobe_efficiency - asis_efficiency, 1)

    return EnrichedKPI(
        name="Eficiencia del Proceso (VAR)",
        unit="%",
        asis_value=asis_efficiency,
        tobe_value=tobe_efficiency,
        reduction_absolute=delta,
        reduction_pct=delta,
        interpretation=(
            f"La eficiencia (Value-Added Ratio) mejora de "
            f"{asis_efficiency:.1f}% a {tobe_efficiency:.1f}% "
            f"(benchmark Lean saludable: > 50%)"
        ),
    )


def _calc_roi(
    asis: Process,
    tobe: TOBEProcess,
    cycle_time_kpi: EnrichedKPI,
    headcount_kpi: EnrichedKPI,
    cost_per_hour_usd: float = 25.0,   # Costo promedio hora-hombre (configurable)
    implementation_cost_usd: float = 5000.0,
) -> tuple[float, float, float]:
    """
    Calcula ROI estimado, payback en meses y ahorro anual en horas.

    Returns:
        (roi_pct, payback_months, annual_saving_hrs)
    """
    # Ahorro por ejecución del proceso
    time_saved_hrs_per_run = cycle_time_kpi.reduction_absolute / 60.0

    # Estimación conservadora: proceso se ejecuta 20 veces/mes
    runs_per_month       = 20
    monthly_saving_hrs   = time_saved_hrs_per_run * runs_per_month
    annual_saving_hrs    = monthly_saving_hrs * 12
    annual_saving_usd    = annual_saving_hrs * cost_per_hour_usd

    roi_pct = round(
        ((annual_saving_usd - implementation_cost_usd)
         / implementation_cost_usd * 100), 1
    ) if implementation_cost_usd > 0 else 0.0

    payback_months = round(
        implementation_cost_usd / (monthly_saving_hrs * cost_per_hour_usd), 1
    ) if (monthly_saving_hrs * cost_per_hour_usd) > 0 else 999.0

    return roi_pct, payback_months, round(annual_saving_hrs, 1)


def _estimate_sigma_level(waste_pct: float) -> float:
    """
    Estima el nivel Sigma del proceso basado en el % de desperdicios.
    Aproximación simplificada para procesos administrativos:
    - >50% desperdicio → ~2 sigma
    - 30-50%           → ~3 sigma
    - 15-30%           → ~4 sigma
    - 5-15%            → ~5 sigma
    - <5%              → ~6 sigma
    """
    if waste_pct > 50:   return 2.0
    if waste_pct > 30:   return 3.0
    if waste_pct > 15:   return 4.0
    if waste_pct > 5:    return 5.0
    return 6.0


# ─────────────────────────────────────────────
# ENRIQUECIMIENTO LLM (capa cualitativa)
# ─────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _enrich_kpis_with_llm(
    #llm: ChatOpenAI,
    llm,
    process: Process,
    tobe: TOBEProcess,
    analysis: WasteAnalysisResult,
    kpis: dict,
) -> dict:
    """
    Llama al LLM para enriquecer los KPIs con interpretaciones de negocio.
    Si falla, el sistema continúa con los KPIs cuantitativos sin enriquecimiento.
    """
    json_schema = json.dumps(
        {
            "executive_summary": "string",
            "kpi_enrichments": {
                "cycle_time":          KPIEnrichment.model_json_schema(),
                "headcount":           KPIEnrichment.model_json_schema(),
                "waste_reduction":     KPIEnrichment.model_json_schema(),
                "automation_coverage": KPIEnrichment.model_json_schema(),
                "process_efficiency":  KPIEnrichment.model_json_schema(),
            }
        },
        ensure_ascii=False,
        indent=2,
    )

    automated_count = sum(
        1 for a in tobe.activities
        if a.status == ActivityStatus.AUTOMATED
    )

    chain = kpi_estimation_prompt | llm | JsonOutputParser()

    return chain.invoke({
        "process_name":        process.name,
        "asis_duration_min":   process.total_duration_min,
        "asis_activity_count": len(process.activities),
        "waste_percentage":    analysis.waste_percentage,
        "waste_time_min":      analysis.total_waste_time_min,
        "tobe_duration_min":   tobe.total_duration_min,
        "tobe_activity_count": len(tobe.activities),
        "automated_count":     automated_count,
        "time_reduction_pct":  kpis["cycle_time"].reduction_pct,
        "kpis_json":           json.dumps(kpis, default=str, ensure_ascii=False, indent=2),
        "json_schema":         json_schema,
    })


def _apply_enrichment(kpi: EnrichedKPI, enrichment_data: dict) -> EnrichedKPI:
    """Aplica el enriquecimiento LLM a un KPI calculado."""
    try:
        kpi.enrichment = KPIEnrichment(**enrichment_data)
    except Exception as e:
        logger.warning(f"No se pudo aplicar enriquecimiento al KPI '{kpi.name}': {e}")
    return kpi


# ─────────────────────────────────────────────
# GENERACIÓN DEL REPORTE COMPLETO
# ─────────────────────────────────────────────

def calculate_kpis(
    process: Process,
    tobe: TOBEProcess,
    analysis: WasteAnalysisResult,
    enrich_with_llm: bool = True,
) -> KPIReportV2:
    """
    Función principal del KPI Calculator.
    1. Calcula todos los KPIs de forma determinística
    2. Estima ROI y nivel Sigma
    3. Enriquece con LLM (opcional)
    4. Retorna el reporte completo
    """
    logger.info(f"Calculando KPIs para: '{process.name}'")

    # ── 1. KPIs cuantitativos (determinísticos) ───────────────────────────
    cycle_time    = _calc_cycle_time(process, tobe)
    headcount     = _calc_headcount(process, tobe)
    waste         = _calc_waste_reduction(analysis, tobe)
    automation    = _calc_automation_coverage(process, tobe, analysis)
    efficiency    = _calc_process_efficiency(process, tobe, analysis)

    # ── 2. ROI y Sigma ────────────────────────────────────────────────────
    roi_pct, payback_months, annual_hrs = _calc_roi(process, tobe, cycle_time, headcount)
    sigma_asis = _estimate_sigma_level(analysis.waste_percentage)
    sigma_tobe = _estimate_sigma_level(
        max(0.0, analysis.waste_percentage - waste.reduction_pct * 0.5)
    )

    logger.info(
        f"KPIs calculados: "
        f"ciclo -{cycle_time.reduction_pct}% | "
        f"muda -{waste.reduction_pct}% | "
        f"automatización {automation.tobe_value}% | "
        f"ROI {roi_pct}% | "
        f"sigma {sigma_asis}→{sigma_tobe}"
    )

    # ── 3. Enriquecimiento LLM ────────────────────────────────────────────
    executive_summary = (
        f"La optimización del proceso '{process.name}' reduce el tiempo de ciclo "
        f"en {cycle_time.reduction_pct:.1f}%, elimina {waste.reduction_pct:.1f}% "
        f"de los desperdicios Lean detectados y eleva la cobertura de automatización "
        f"al {automation.tobe_value:.1f}%. El ROI estimado es {roi_pct:.1f}% "
        f"con recuperación de inversión en {payback_months:.1f} meses."
    )

    if enrich_with_llm:
        try:
            #llm = ChatOpenAI(
            #    model=settings.openai_model,
            #    temperature=0.2,
            #    api_key=settings.openai_api_key,
            #)
            llm = get_llm(temperature=0.2)
            
            kpis_dict = {
                "cycle_time":          cycle_time.model_dump(),
                "headcount":           headcount.model_dump(),
                "waste_reduction":     waste.model_dump(),
                "automation_coverage": automation.model_dump(),
                "process_efficiency":  efficiency.model_dump(),
            }
            enrichment = _enrich_kpis_with_llm(
                llm, process, tobe, analysis, kpis_dict
            )

            # Aplicar enriquecimiento a cada KPI
            enrichments = enrichment.get("kpi_enrichments", {})
            cycle_time = _apply_enrichment(cycle_time, enrichments.get("cycle_time", {}))
            headcount  = _apply_enrichment(headcount,  enrichments.get("headcount", {}))
            waste      = _apply_enrichment(waste,      enrichments.get("waste_reduction", {}))
            automation = _apply_enrichment(automation, enrichments.get("automation_coverage", {}))
            efficiency = _apply_enrichment(efficiency, enrichments.get("process_efficiency", {}))

            executive_summary = enrichment.get("executive_summary", executive_summary)
            logger.info("KPIs enriquecidos con LLM exitosamente")

        except Exception as e:
            # El enriquecimiento no es bloqueante
            logger.warning(
                f"Enriquecimiento LLM falló — usando KPIs sin enriquecimiento: {e}"
            )

    # ── 4. Construir reporte ──────────────────────────────────────────────
    return KPIReportV2(
        process_id=process.id,
        tobe_process_id=tobe.id,
        cycle_time=cycle_time,
        headcount=headcount,
        waste_reduction=waste,
        automation_coverage=automation,
        process_efficiency=efficiency,
        estimated_roi_pct=roi_pct,
        estimated_payback_months=payback_months,
        estimated_annual_saving_hrs=annual_hrs,
        executive_summary=executive_summary,
        sigma_level_asis=sigma_asis,
        sigma_level_tobe=sigma_tobe,
    )


# ─────────────────────────────────────────────
# NODO LANGGRAPH
# ─────────────────────────────────────────────

def node_calculate_kpis(state: AgentState) -> dict:
    """
    Nodo LangGraph: calcula el reporte completo de KPIs AS-IS vs TO-BE.

    Entrada del estado: asis_process, tobe_process, waste_analysis, bpmn_ok
    Salida al estado:   kpi_report, kpi_ok, errors
    """
    logger.info("── Nodo: calculate_kpis ──")

    process  = state.asis_process
    tobe     = state.tobe_process
    analysis = state.waste_analysis

    if not all([process, tobe, analysis]):
        missing = [
            name for name, val in [
                ("asis_process",   process),
                ("tobe_process",   tobe),
                ("waste_analysis", analysis),
            ] if val is None
        ]
        return {
            "kpi_ok": False,
            "errors": state.errors + [
                f"calculate_kpis: faltan campos requeridos: {missing}"
            ],
            "current_node": "calculate_kpis",
        }

    try:
        report = calculate_kpis(
            process=process,
            tobe=tobe,
            analysis=analysis,
            enrich_with_llm=True,
        )

        # Persistir el caso en la Vector DB para futuros retrievals RAG
        _persist_case_to_rag(process, tobe, report)

        return {
            "kpi_report":   report,
            "kpi_ok":       True,
            "current_node": "calculate_kpis",
        }

    except Exception as e:
        logger.error(f"Error en calculate_kpis: {e}")
        return {
            "kpi_ok": False,
            "errors": state.errors + [f"calculate_kpis: {str(e)}"],
            "current_node": "calculate_kpis",
        }


# ─────────────────────────────────────────────
# PERSISTENCIA RAG POST-ANÁLISIS
# ─────────────────────────────────────────────

def _persist_case_to_rag(
    process: Process,
    tobe: TOBEProcess,
    report: KPIReportV2,
) -> None:
    """
    Persiste el caso completo en la Vector DB al finalizar el análisis.
    Enriquece la base de conocimiento para futuros procesos similares.
    """
    try:
        from rag.vector_store import store_process_case

        asis_summary = (
            f"Proceso: {process.name}. "
            f"Duración: {process.total_duration_min:.0f} min. "
            f"Actividades: {len(process.activities)}. "
            f"Área: {process.owner}."
        )
        tobe_summary = (
            f"Reducción tiempo: {report.cycle_time.reduction_pct:.1f}%. "
            f"Muda eliminada: {report.waste_reduction.reduction_pct:.1f}%. "
            f"Automatización: {report.automation_coverage.tobe_value:.1f}%. "
            f"ROI: {report.estimated_roi_pct:.1f}%."
        )
        improvements = [
            f"Tiempo de ciclo reducido en {report.cycle_time.reduction_absolute:.0f} min",
            f"Nivel Sigma mejorado de {report.sigma_level_asis} a {report.sigma_level_tobe}",
            f"Ahorro anual estimado: {report.estimated_annual_saving_hrs:.0f} horas",
        ]

        store_process_case(
            process_name=process.name,
            asis_summary=asis_summary,
            tobe_summary=tobe_summary,
            industry=process.owner,
            improvements=improvements,
        )
        logger.info(f"Caso '{process.name}' persistido en Vector DB RAG")

    except Exception as e:
        # No bloquea — el RAG es opcional
        logger.warning(f"No se pudo persistir el caso en RAG: {e}")