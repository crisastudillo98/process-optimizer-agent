"""
Tests de integración del pipeline completo AS-IS → TO-BE.
Ejecutar con: pytest tests/test_integration_pipeline.py -v -m integration

Requieren:
    - OPENAI_API_KEY válida en .env
    - ChromaDB corriendo (o modo local)
"""
import pytest
from agent.orchestrator import build_graph
from models.schemas import AgentState


PROCESS_FACTURACION = """
El proceso de facturación inicia cuando el área comercial notifica al equipo
de finanzas sobre un pedido aprobado. Un asistente administrativo descarga
manualmente el pedido del sistema CRM (15 min), verifica los datos del cliente
en una hoja de Excel (20 min) y los compara con el sistema ERP para detectar
inconsistencias (30 min). Si hay inconsistencias, las reporta por correo al
área comercial y espera respuesta hasta 2 días hábiles (2880 min).
Una vez validados los datos, el asistente genera la factura manualmente en SAP (25 min),
la imprime y escanea para archivarla en una carpeta de red (15 min),
y la envía por correo al cliente (10 min). Finalmente, registra el envío
en la hoja de control de Excel (10 min) y espera la confirmación de recepción
del cliente hasta 3 días hábiles (4320 min). El proceso lo ejecutan 2 personas
del área de finanzas y afecta directamente los tiempos de cobro.
"""

PROCESS_CONTRATACION = """
El proceso de contratación de personal inicia cuando un jefe de área
solicita una vacante vía correo electrónico al área de RRHH (sin sistema).
Un reclutador recibe el correo, crea manualmente la descripción del cargo
en Word (45 min) y la publica en portales de empleo (LinkedIn, Computrabajo)
de forma manual (60 min). Espera recepción de hojas de vida durante 10 días
hábiles (14400 min). Revisa cada HV manualmente en Excel (120 min por proceso),
preselecciona candidatos y les envía pruebas técnicas por correo (30 min).
Espera que los candidatos respondan hasta 5 días hábiles (7200 min).
Agenda entrevistas coordinando por correo con el jefe de área (60 min),
realiza las entrevistas (90 min por candidato, promedio 5 candidatos = 450 min),
consolida resultados en Excel (60 min) y presenta informe al jefe de área (30 min).
Una vez seleccionado el candidato, inicia el proceso de contratación con
documentación física (180 min). El proceso no tiene sistema de seguimiento
y toda la información vive en correos y archivos Excel personales.
"""


@pytest.mark.integration
class TestPipelineIntegration:

    @pytest.fixture
    def graph(self):
        return build_graph()

    def test_full_pipeline_facturacion(self, graph):
        """Pipeline completo con proceso de facturación real."""
        state = AgentState(
            raw_input=PROCESS_FACTURACION,
            current_node="start",
        )

        result = graph.invoke(
            state.model_dump(),
            config={"configurable": {"thread_id": "test-facturacion"}},
        )
        final = AgentState(**result)

        # Extracción
        assert final.extraction_ok is True
        assert final.asis_process is not None
        assert len(final.asis_process.activities) >= 5
        assert final.asis_process.total_duration_min > 0

        # Análisis Lean
        assert final.analysis_ok is True
        assert final.waste_analysis is not None
        assert final.waste_analysis.waste_count >= 1
        assert final.waste_analysis.waste_percentage > 0

        # TO-BE
        assert final.optimization_ok is True
        assert final.tobe_process is not None
        assert final.tobe_process.total_duration_min < final.asis_process.total_duration_min
        assert final.tobe_process.sipoc is not None

        # BPMN
        assert final.bpmn_ok is True
        assert final.bpmn_output is not None
        assert final.bpmn_output.element_count > 0
        assert "<?xml" in final.bpmn_output.xml_content

        # KPIs
        assert final.kpi_ok is True
        assert final.kpi_report is not None
        assert final.kpi_report.cycle_time.reduction_pct > 0
        assert final.kpi_report.estimated_roi_pct is not None
        assert final.kpi_report.sigma_level_asis is not None
        assert final.kpi_report.executive_summary != ""

        # Sin errores críticos
        critical_errors = [
            e for e in final.errors
            if not e.startswith("⚠️")    # advertencias son ok
        ]
        assert len(critical_errors) == 0

    def test_full_pipeline_contratacion(self, graph):
        """Pipeline completo con proceso de contratación real."""
        state = AgentState(
            raw_input=PROCESS_CONTRATACION,
            current_node="start",
        )

        result = graph.invoke(
            state.model_dump(),
            config={"configurable": {"thread_id": "test-contratacion"}},
        )
        final = AgentState(**result)

        assert final.extraction_ok is True
        assert final.kpi_ok is True

        # Proceso de contratación tiene muchas esperas → alta Muda
        assert final.waste_analysis.waste_percentage >= 30.0

        # Debe detectar esperas como Muda principal
        waste_types = [wt.value for wt in final.waste_analysis.main_waste_types]
        assert "espera" in waste_types

        # Quick wins de Kaizen
        assert len(final.waste_analysis.kaizen_quick_wins) >= 2

    def test_pipeline_tobe_is_shorter(self, graph):
        """El TO-BE siempre debe ser más corto que el AS-IS."""
        state = AgentState(raw_input=PROCESS_FACTURACION)
        result = graph.invoke(state.model_dump())
        final  = AgentState(**result)

        assert final.tobe_process.total_duration_min \
             < final.asis_process.total_duration_min

    def test_pipeline_bpmn_is_valid_xml(self, graph):
        """El BPMN generado debe ser XML parseable y válido."""
        from lxml import etree
        from agent.bpmn_generator import _validate_bpmn_xml

        state  = AgentState(raw_input=PROCESS_FACTURACION)
        result = graph.invoke(state.model_dump())
        final  = AgentState(**result)

        errors = _validate_bpmn_xml(final.bpmn_output.xml_content)
        assert errors == [], f"Errores BPMN: {errors}"

    def test_pipeline_rag_enriches_context(self, graph):
        """
        Después de ejecutar el primer proceso, el segundo debe
        encontrar contexto RAG (el primero queda en la Vector DB).
        """
        # Primer proceso
        state1 = AgentState(raw_input=PROCESS_FACTURACION)
        graph.invoke(state1.model_dump())

        # Segundo proceso similar — debe recuperar contexto del primero
        state2 = AgentState(raw_input=PROCESS_CONTRATACION)
        result = graph.invoke(state2.model_dump())
        final  = AgentState(**result)

        # El contexto RAG no debe estar vacío
        assert final.rag_context
        assert any(
            "Sin contexto" not in chunk
            for chunk in final.rag_context
        )