"""
Script de inicialización de la knowledge base Lean/Six Sigma/Kaizen.
Ejecutar UNA SOLA VEZ antes de arrancar el sistema:
    python -m rag.seed_knowledge
"""
from langchain_core.documents import Document
from rag.vector_store import store_lean_knowledge
from observability.logger import get_logger

logger = get_logger(__name__)

LEAN_KNOWLEDGE_BASE = [
    Document(
        page_content=(
            "MUDA ESPERA: Las esperas son el desperdicio más común en procesos administrativos. "
            "Soluciones Lean: implementar aprobaciones digitales con workflows automáticos "
            "(Power Automate, n8n), establecer SLAs de respuesta con alertas automáticas, "
            "usar sistemas de notificación en tiempo real. "
            "Kaizen quick win: mapa de flujo de valor para identificar todos los puntos de espera "
            "y su duración real. Six Sigma: medir variabilidad en tiempos de aprobación con "
            "gráficas de control para establecer límites de control UCL/LCL."
        ),
        metadata={"type": "lean_pattern", "waste_type": "espera", "methodology": "Lean"},
    ),
    Document(
        page_content=(
            "MUDA SOBREPROCESO: Actividades que agregan más de lo que el cliente necesita. "
            "Síntomas: múltiples revisiones del mismo documento, formularios con campos innecesarios, "
            "reportes que nadie lee. Soluciones: Value Stream Mapping para identificar pasos sin valor, "
            "técnica 5W1H para cuestionar cada actividad, eliminación de aprobaciones redundantes "
            "mediante matriz RACI. Six Sigma DMAIC: medir el costo del sobreproceso en horas-hombre, "
            "analizar causas raíz con diagrama Ishikawa."
        ),
        metadata={"type": "lean_pattern", "waste_type": "sobreproceso", "methodology": "Lean"},
    ),
    Document(
        page_content=(
            "MUDA DEFECTOS Y REPROCESOS: Errores que requieren corrección y consumen doble recurso. "
            "Soluciones: implementar Poka-Yoke (pruebas de error) en formularios y sistemas, "
            "validaciones automáticas en el punto de captura, checklists digitales, "
            "automatización de verificaciones con RPA para eliminar errores de digitación. "
            "Six Sigma: analizar la tasa de defectos (DPMO), aplicar control estadístico de proceso (SPC). "
            "Kaizen: sesiones de 5 Porqués para identificar causa raíz de cada tipo de defecto."
        ),
        metadata={"type": "lean_pattern", "waste_type": "defectos", "methodology": "Lean"},
    ),
    Document(
        page_content=(
            "AUTOMATIZACIÓN RPA: Las actividades operativas repetitivas son candidatas directas a RPA. "
            "Criterios de automatización: alto volumen, reglas claras, datos estructurados, "
            "baja variabilidad, múltiples sistemas involucrados. "
            "Herramientas: UiPath para procesos complejos con UI, Power Automate para ecosistema Microsoft, "
            "n8n para integraciones API-first, Python + Selenium para procesos web. "
            "ROI típico de RPA: reducción 60-80% del tiempo de ciclo, "
            "eliminación de errores humanos en >90% de los casos."
        ),
        metadata={"type": "automation_pattern", "methodology": "RPA"},
    ),
    Document(
        page_content=(
            "KAIZEN QUICK WINS: Mejoras de alto impacto y baja inversión implementables en < 2 semanas. "
            "Ejemplos por categoría: "
            "1. Digitalización: reemplazar formularios en papel por formularios web (Google Forms, Typeform). "
            "2. Comunicación: crear canales dedicados en Slack/Teams por proceso para reducir correos. "
            "3. Visibilidad: tableros Kanban digitales (Trello, Jira) para seguimiento de tareas. "
            "4. Estandarización: SOPs digitales con Notion o Confluence para eliminar variabilidad. "
            "5. Automatización liviana: alertas automáticas por correo/SMS para hitos del proceso."
        ),
        metadata={"type": "kaizen_pattern", "methodology": "Kaizen"},
    ),
    Document(
        page_content=(
            "SIX SIGMA DMAIC PARA OPTIMIZACIÓN DE PROCESOS: "
            "D (Definir): identificar el proceso, sus clientes y sus CTQs (Critical to Quality). "
            "M (Medir): documentar el AS-IS, medir tiempos de ciclo, tasas de error y capacidad. "
            "A (Analizar): identificar causas raíz de desperdicios con Pareto, Ishikawa y ANOVA. "
            "I (Mejorar): diseñar el TO-BE, pilotar cambios, validar mejoras con datos. "
            "C (Controlar): implementar controles (SPC, dashboards) para sostener las mejoras. "
            "KPIs clave: tiempo de ciclo, tasa de defectos (DPMO), costo por transacción, NPS."
        ),
        metadata={"type": "methodology", "methodology": "Six Sigma"},
    ),
    Document(
        page_content=(
            "VALUE STREAM MAPPING (VSM): Herramienta Lean para visualizar el flujo de valor completo. "
            "Símbolos clave: caja de proceso (actividad), triángulo (inventario/espera), "
            "flecha push/pull, timeline de valor agregado vs no agregado. "
            "Métricas VSM: Lead Time total, Process Time (tiempo real de trabajo), "
            "eficiencia del proceso = Process Time / Lead Time * 100. "
            "Procesos típicos tienen eficiencia < 20% — el 80% es espera. "
            "El VSM TO-BE debe apuntar a eficiencia > 50% mediante eliminación de esperas "
            "y automatización de actividades de bajo valor."
        ),
        metadata={"type": "tool_pattern", "methodology": "Lean", "tool": "VSM"},
    ),
    Document(
        page_content=(
            "MATRIZ SIPOC PARA PROCESOS TO-BE: "
            "Suppliers (proveedores del proceso), Inputs (entradas), Process (actividades clave), "
            "Outputs (salidas), Customers (clientes del proceso). "
            "El SIPOC del TO-BE debe reflejar: eliminación de entradas innecesarias, "
            "reducción de proveedores de información redundantes, "
            "outputs claramente definidos y medibles, clientes internos/externos identificados. "
            "Uso en IA: el agente debe generar automáticamente el SIPOC del proceso optimizado "
            "para facilitar la comunicación con stakeholders no técnicos."
        ),
        metadata={"type": "tool_pattern", "methodology": "Six Sigma", "tool": "SIPOC"},
    ),
]


def seed() -> None:
    logger.info(f"Inicializando knowledge base con {len(LEAN_KNOWLEDGE_BASE)} documentos...")
    store_lean_knowledge(LEAN_KNOWLEDGE_BASE)
    logger.info("✅ Knowledge base Lean/Six Sigma/Kaizen inicializada correctamente")


if __name__ == "__main__":
    seed()