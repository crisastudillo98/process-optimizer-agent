# Todos los modulos del agente se comunican a través de estos modelos
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class ActivityType(str, Enum):
    """Clasificación funcional de la actividad."""
    OPERATIVE  = "operativa"    # Ejecución física o manual
    ANALYTICAL = "analitica"    # Procesamiento de información
    COGNITIVE  = "cognitiva"    # Juicio, decisión, razonamiento


class WasteType(str, Enum):
    """7 Mudas Lean + la octava (talento no utilizado)."""
    WAITING           = "espera"
    OVERPROCESSING    = "sobreproceso"
    DEFECTS           = "defectos"
    OVERPRODUCTION    = "sobreproduccion"
    TRANSPORT         = "transporte"
    INVENTORY         = "inventario"
    MOTION            = "movimiento"
    UNUSED_TALENT     = "talento_no_utilizado"


class ActivityStatus(str, Enum):
    """Estado de la actividad en el proceso TO-BE."""
    KEPT        = "conservada"
    OPTIMIZED   = "optimizada"
    AUTOMATED   = "automatizada"
    ELIMINATED  = "eliminada"
    COMBINED    = "combinada"
    NEW         = "nueva"


class WasteClassification(str, Enum):
    """Resultado del análisis de valor de una actividad."""
    VALUE_ADDED     = "genera_valor"
    WASTE           = "desperdicio"
    NEEDS_INFO      = "requiere_informacion"


# ─────────────────────────────────────────────
# BLOQUES BASE
# ─────────────────────────────────────────────

class SubActivity(BaseModel):
    """Descomposición atómica de una actividad compleja."""
    id: str                  = Field(..., description="ID único: SA-001")
    name: str                = Field(..., description="Nombre de la subactividad")
    description: str         = Field(..., description="Descripción detallada")
    type: ActivityType       = Field(..., description="Tipo funcional")
    estimated_duration_min: Optional[float] = Field(None, description="Duración estimada en minutos")
    is_automatable: bool     = Field(default=False, description="¿Es automatizable con RPA o IA?")
    automation_tool: Optional[str] = Field(None, description="Herramienta sugerida si es automatizable")


class Activity(BaseModel):
    """Actividad individual dentro del proceso AS-IS."""
    id: str                  = Field(..., description="ID único: ACT-001")
    name: str                = Field(..., description="Nombre de la actividad")
    description: str         = Field(..., description="Descripción detallada")
    responsible: str         = Field(..., description="Rol o actor responsable")
    type: ActivityType       = Field(..., description="Clasificación funcional")
    estimated_duration_min: float  = Field(..., description="Duración estimada en minutos")
    depends_on: list[str]    = Field(default_factory=list, description="IDs de actividades previas")
    systems_used: list[str]  = Field(default_factory=list, description="Sistemas o herramientas involucradas")
    subactivities: list[SubActivity] = Field(default_factory=list)

    # Análisis de desperdicios
    waste_classification: Optional[WasteClassification] = None
    waste_type: Optional[WasteType]                     = None
    waste_justification: Optional[str]                  = None


class Process(BaseModel):
    """Representación estructurada del proceso AS-IS."""
    id: str                  = Field(..., description="ID único del proceso")
    name: str                = Field(..., description="Nombre del proceso")
    description: str         = Field(..., description="Descripción general del proceso")
    owner: str               = Field(..., description="Área o responsable del proceso")
    scope: str               = Field(..., description="Alcance: inicio y fin del proceso")
    participants: list[str]  = Field(default_factory=list, description="Roles involucrados")
    systems: list[str]       = Field(default_factory=list, description="Sistemas de soporte")
    activities: list[Activity] = Field(default_factory=list)
    total_duration_min: float  = Field(default=0.0, description="Duración total estimada")
    raw_input: str             = Field(default="", description="Texto original ingresado por el usuario")
    extracted_at: datetime     = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# ANÁLISIS DE DESPERDICIOS
# ─────────────────────────────────────────────

class WasteAnalysis(BaseModel):
    """Resultado del análisis Lean sobre el proceso AS-IS."""
    process_id: str
    total_activities: int
    value_added_count: int
    waste_count: int
    needs_info_count: int
    waste_percentage: float        = Field(..., description="% de actividades que son desperdicio")
    main_waste_types: list[WasteType] = Field(default_factory=list)
    analyzed_activities: list[Activity] = Field(default_factory=list)
    summary: str                   = Field(..., description="Resumen ejecutivo del análisis")
    analyzed_at: datetime          = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# PROCESO TO-BE
# ─────────────────────────────────────────────

class OptimizedActivity(BaseModel):
    """Actividad en el proceso TO-BE con propuesta de mejora."""
    id: str                      = Field(..., description="ID: OPT-001")
    original_activity_id: Optional[str] = Field(None, description="Referencia al AS-IS original")
    name: str
    description: str
    responsible: str
    type: ActivityType
    status: ActivityStatus       = Field(..., description="Qué se hace con esta actividad")
    estimated_duration_min: float
    duration_reduction_pct: float = Field(default=0.0, description="% de reducción vs AS-IS")
    is_automatable: bool          = Field(default=False)
    automation_tool: Optional[str] = None
    improvement_justification: str = Field(..., description="Por qué se aplica este cambio")
    depends_on: list[str]          = Field(default_factory=list)


class TOBEProcess(BaseModel):
    """Propuesta de proceso optimizado."""
    id: str
    original_process_id: str
    name: str
    description: str
    owner: str
    activities: list[OptimizedActivity] = Field(default_factory=list)
    total_duration_min: float
    generated_at: datetime              = Field(default_factory=datetime.utcnow)
    applied_methodologies: list[str]    = Field(
        default=["Lean", "Six Sigma", "Kaizen"],
        description="Marcos aplicados en la optimización"
    )
    sipoc: Optional[dict]               = Field(None, description="Suppliers, Inputs, Process, Outputs, Customers")
    human_approved: bool                = Field(default=False, description="¿Validado por experto humano?")
    approver_notes: Optional[str]       = None


# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────

class KPI(BaseModel):
    """Métrica cuantitativa individual."""
    name: str
    unit: str
    asis_value: float
    tobe_value: float
    reduction_absolute: float
    reduction_pct: float
    interpretation: str


class KPIReport(BaseModel):
    """Conjunto de KPIs del proceso optimizado."""
    process_id: str
    tobe_process_id: str
    cycle_time_reduction: KPI
    headcount_reduction: KPI
    waste_reduction: KPI
    automation_coverage: KPI      # % de actividades automatizables
    estimated_roi_pct: Optional[float] = Field(None, description="ROI estimado del proyecto de mejora")
    executive_summary: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# MODELOS DEL ANALYZER (Fase 3)
# ─────────────────────────────────────────────

class ActivityWasteDetail(BaseModel):
    """Resultado del análisis de una actividad individual."""
    activity_id: str
    activity_name: str
    waste_classification: WasteClassification
    waste_type: Optional[WasteType]       = None
    waste_justification: str              = Field(..., description="Explicación detallada")
    estimated_waste_time_min: float       = Field(
        default=0.0,
        description="Minutos de desperdicio estimados en esta actividad"
    )
    is_automatable: bool                  = Field(default=False)
    automation_tool: Optional[str]        = None
    automation_justification: Optional[str] = None


class Redundancy(BaseModel):
    """Par de actividades redundantes o solapadas."""
    activity_ids: list[str]       = Field(..., description="IDs de las actividades redundantes")
    activity_names: list[str]     = Field(..., description="Nombres de las actividades redundantes")
    redundancy_type: str          = Field(
        ...,
        description="Tipo: 'duplicacion', 'solapamiento', 'revision_multiple', 'transferencia_repetida'"
    )
    description: str              = Field(..., description="Descripción del solapamiento")
    suggested_action: str         = Field(..., description="Qué hacer: eliminar, combinar, etc.")


class WasteAnalysisResult(BaseModel):
    """
    Resultado completo del análisis Lean del proceso AS-IS.
    Reemplaza y extiende el modelo WasteAnalysis de Fase 1.
    """
    process_id: str
    process_name: str

    # Detalle por actividad
    activity_details: list[ActivityWasteDetail] = Field(default_factory=list)

    # Redundancias detectadas
    redundancies: list[Redundancy]              = Field(default_factory=list)

    # Totales calculados
    total_activities: int                       = 0
    value_added_count: int                      = 0
    waste_count: int                            = 0
    needs_info_count: int                       = 0
    waste_percentage: float                     = 0.0
    total_waste_time_min: float                 = 0.0
    automatable_count: int                      = 0
    automation_coverage_pct: float              = 0.0

    # Tipos de Muda predominantes
    main_waste_types: list[WasteType]           = Field(default_factory=list)

    # Resúmenes ejecutivos
    lean_summary: str      = Field(..., description="Resumen Lean del análisis")
    six_sigma_insights: str = Field(..., description="Insights Six Sigma: variabilidad y defectos")
    kaizen_quick_wins: list[str] = Field(
        default_factory=list,
        description="Mejoras rápidas (quick wins) aplicables inmediatamente"
    )

    analyzed_at: datetime  = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# MODELOS BPMN (Fase 5)
# ─────────────────────────────────────────────

class BPMNElementType(str, Enum):
    """Tipos de elementos BPMN 2.0 soportados."""
    START_EVENT        = "startEvent"
    END_EVENT          = "endEvent"
    TASK               = "task"
    USER_TASK          = "userTask"
    SERVICE_TASK       = "serviceTask"
    EXCLUSIVE_GATEWAY  = "exclusiveGateway"
    PARALLEL_GATEWAY   = "parallelGateway"
    SEQUENCE_FLOW      = "sequenceFlow"


class BPMNElement(BaseModel):
    """Elemento individual del diagrama BPMN."""
    id: str                        = Field(..., description="ID único: task_1, gateway_1...")
    type: BPMNElementType
    name: str                      = Field(..., description="Etiqueta del elemento (máx 50 chars)")
    lane: Optional[str]            = Field(None, description="Carril/responsable del elemento")
    documentation: Optional[str]  = Field(None, description="Descripción detallada del elemento")


class BPMNSequenceFlow(BaseModel):
    """Flujo de secuencia entre dos elementos BPMN."""
    id: str           = Field(..., description="ID único: flow_1, flow_2...")
    source_ref: str   = Field(..., description="ID del elemento origen")
    target_ref: str   = Field(..., description="ID del elemento destino")
    name: Optional[str] = Field(None, description="Condición del flujo (para gateways)")


class BPMNLane(BaseModel):
    """Carril (lane) que agrupa actividades por responsable."""
    id: str
    name: str                          = Field(..., description="Nombre del responsable o rol")
    element_ids: list[str]             = Field(default_factory=list)


class BPMNStructure(BaseModel):
    """
    Estructura lógica intermedia del proceso BPMN.
    El LLM genera esto; luego el generador XML lo convierte a BPMN 2.0.
    """
    process_id: str
    process_name: str
    lanes: list[BPMNLane]              = Field(default_factory=list)
    elements: list[BPMNElement]        = Field(default_factory=list)
    sequence_flows: list[BPMNSequenceFlow] = Field(default_factory=list)


class BPMNOutput(BaseModel):
    """Resultado final del generador BPMN."""
    process_id: str
    process_name: str
    xml_content: str       = Field(..., description="XML BPMN 2.0 completo y válido")
    file_path: str         = Field(..., description="Ruta donde fue guardado el archivo .bpmn")
    element_count: int     = Field(..., description="Total de elementos en el diagrama")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# MODELOS KPI EXTENDIDOS (Fase 6)
# ─────────────────────────────────────────────

class KPIEnrichment(BaseModel):
    """Capa cualitativa de un KPI — generada por LLM."""
    business_interpretation: str  = Field(
        ..., description="Qué significa este número para el negocio"
    )
    industry_benchmark: str       = Field(
        ..., description="Comparación con estándares Lean/Six Sigma de la industria"
    )
    implementation_risk: str      = Field(
        ..., description="Qué podría impedir alcanzar esta mejora"
    )
    next_step: str                = Field(
        ..., description="Acción concreta para materializar este KPI"
    )


class EnrichedKPI(BaseModel):
    """KPI cuantitativo + interpretación cualitativa LLM."""
    name: str
    unit: str
    asis_value: float
    tobe_value: float
    reduction_absolute: float
    reduction_pct: float
    interpretation: str           # Resumen corto (calculado)
    enrichment: Optional[KPIEnrichment] = None   # Detalle LLM


class KPIReportV2(BaseModel):
    """
    Reporte KPI completo con métricas cuantitativas + enriquecimiento LLM.
    Reemplaza KPIReport de Fase 1.
    """
    process_id: str
    tobe_process_id: str

    # KPIs principales
    cycle_time:          EnrichedKPI
    headcount:           EnrichedKPI
    waste_reduction:     EnrichedKPI
    automation_coverage: EnrichedKPI
    process_efficiency:  EnrichedKPI   # Process Time / Lead Time * 100

    # ROI
    estimated_roi_pct:          Optional[float] = None
    estimated_payback_months:   Optional[float] = None
    estimated_annual_saving_hrs: Optional[float] = None

    # Resumen
    executive_summary: str
    sigma_level_asis:  Optional[float] = Field(
        None, description="Nivel Sigma estimado del proceso AS-IS"
    )
    sigma_level_tobe:  Optional[float] = Field(
        None, description="Nivel Sigma estimado del proceso TO-BE"
    )

    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# ESTADO COMPARTIDO DEL GRAFO (LangGraph)
# ─────────────────────────────────────────────

class AgentState(BaseModel):
    """
    Estado central del StateGraph de LangGraph.
    Cada nodo lee y escribe sobre esta estructura.
    """
    # Input
    raw_input: str                        = Field(default="", description="Texto o ruta de archivo ingresado")
    input_file_path: Optional[str]        = None

    # Nodo 1: Process Extractor
    asis_process: Optional[Process]       = None
    extraction_ok: bool                   = False

    # Nodo 2: Analyzer
    #waste_analysis: Optional[WasteAnalysis] = None
    waste_analysis: Optional[WasteAnalysisResult] = None
    analysis_ok: bool                       = False

    # Nodo 3: RAG context
    rag_context: list[str]                = Field(default_factory=list, description="Fragmentos recuperados")

    # Nodo 4: Optimizer
    tobe_process: Optional[TOBEProcess]   = None
    optimization_ok: bool                 = False

    # Nodo 5: HITL
    hitl_required: bool                   = False
    hitl_approved: bool                   = False
    hitl_feedback: Optional[str]          = None
    hitl_retries: int                     = Field(default=0, description="Contador de re-optimizaciones HITL")

    # Nodo 6: BPMN Generator
    #bpmn_xml: Optional[str]               = None
    #bpmn_file_path: Optional[str]         = None
    #bpmn_ok: bool                         = False
    bpmn_output: Optional[BPMNOutput] = None
    bpmn_ok: bool                     = False

    # Nodo 7: KPI Calculator
    #kpi_report: Optional[KPIReport]       = None
    kpi_report: Optional[KPIReportV2] = None
    kpi_ok: bool                          = False

    # Control de errores
    errors: list[str]                     = Field(default_factory=list)
    current_node: str                     = Field(default="start")

    model_config = {"arbitrary_types_allowed": True}