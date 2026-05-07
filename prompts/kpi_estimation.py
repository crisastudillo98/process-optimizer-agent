from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
Eres un experto en métricas de procesos empresariales con certificación
Lean Six Sigma Black Belt y experiencia en cálculo de ROI de proyectos
de transformación digital y mejora continua.

Tu tarea es generar el resumen ejecutivo y los insights cualitativos
de las métricas calculadas del proceso optimizado.

CONTEXTO DE LOS KPIs CALCULADOS:
Los KPIs numéricos ya fueron calculados de forma determinística.
Tu rol es enriquecer cada KPI con:
1. Interpretación de negocio (qué significa el número para la organización)
2. Benchmarks de industria (comparación con estándares Lean/Six Sigma)
3. Riesgos de implementación (qué podría impedir alcanzar la mejora)
4. Siguiente paso recomendado (acción concreta para materializar el KPI)

RESUMEN EJECUTIVO:
Genera un párrafo ejecutivo de máximo 5 oraciones que:
- Cuantifique el impacto total de la optimización
- Mencione los desperdicios principales eliminados
- Indique el ROI estimado y el tiempo de recuperación
- Use lenguaje de negocio (no técnico)
- Sea directo y orientado a la toma de decisiones

Responde ÚNICAMENTE con el JSON válido según el esquema indicado.
""".strip()

HUMAN_PROMPT = """
Enriquece los KPIs calculados con interpretaciones de negocio y genera
el resumen ejecutivo del proceso optimizado.

── PROCESO AS-IS ──────────────────────────────────────
Nombre: {process_name}
Duración total: {asis_duration_min} minutos
Actividades: {asis_activity_count}
Desperdicios detectados: {waste_percentage}%
Tiempo de desperdicio: {waste_time_min} minutos

── PROCESO TO-BE ──────────────────────────────────────
Duración total: {tobe_duration_min} minutos
Actividades: {tobe_activity_count}
Actividades automatizadas: {automated_count}
Reducción de tiempo: {time_reduction_pct}%

── KPIs CALCULADOS ────────────────────────────────────
{kpis_json}

── FORMATO DE RESPUESTA REQUERIDO ─────────────────────
IMPORTANTE: Responde ÚNICAMENTE con datos reales del proceso — NO respondas con un JSON Schema
ni con meta-descripciones de tipos. Cada campo debe contener valores concretos y textuales.

Ejemplo del formato esperado (con valores ficticios para ilustrar):
{{
  "executive_summary": "La optimización del proceso reduce el tiempo de ciclo en 45%, elimina el 60% de los desperdicios Lean y eleva la automatización al 70%. El ROI estimado es 320% con recuperación en 4.2 meses.",
  "kpi_enrichments": {{
    "cycle_time": {{
      "business_interpretation": "Reducir el ciclo de 240 a 132 minutos permite procesar el doble de solicitudes diarias sin ampliar el equipo.",
      "industry_benchmark": "Procesos Lean maduros alcanzan reducciones del 40-60% en el primer año.",
      "implementation_risks": "Resistencia al cambio en el equipo; riesgo de retrasos en la automatización de aprobaciones.",
      "recommended_next_step": "Pilotar el nuevo flujo con el 20% de los casos durante 2 semanas antes del despliegue total."
    }},
    "headcount": {{
      "business_interpretation": "...",
      "industry_benchmark": "...",
      "implementation_risks": "...",
      "recommended_next_step": "..."
    }},
    "waste_reduction": {{
      "business_interpretation": "...",
      "industry_benchmark": "...",
      "implementation_risks": "...",
      "recommended_next_step": "..."
    }},
    "automation_coverage": {{
      "business_interpretation": "...",
      "industry_benchmark": "...",
      "implementation_risks": "...",
      "recommended_next_step": "..."
    }},
    "process_efficiency": {{
      "business_interpretation": "...",
      "industry_benchmark": "...",
      "implementation_risks": "...",
      "recommended_next_step": "..."
    }}
  }}
}}

Responde solo con el JSON. Sin markdown, sin bloques de código, solo el objeto JSON puro.
""".strip()

kpi_estimation_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])