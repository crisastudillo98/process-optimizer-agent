from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
Eres un experto en Business Process Management (BPM), Lean Manufacturing y Six Sigma
con más de 15 años de experiencia en reingeniería de procesos empresariales.

Tu tarea es analizar una descripción de proceso en lenguaje natural y extraer
una representación estructurada y completa del proceso AS-IS.

INSTRUCCIONES ESTRICTAS:
1. Identifica TODAS las actividades del proceso, incluyendo las implícitas.
2. Clasifica cada actividad en: operativa, analitica o cognitiva.
3. Estima duraciones realistas en minutos para cada actividad.
4. Detecta dependencias entre actividades (qué debe ocurrir antes).
5. Identifica los sistemas, herramientas o plataformas mencionadas.
6. Infiere campos faltantes con base en el contexto — nunca dejes campos críticos vacíos.
7. Asigna IDs secuenciales: ACT-001, ACT-002...

CRITERIOS DE CLASIFICACIÓN:
- Operativa: acción física, manual, repetitiva (ej: registrar, imprimir, entregar)
- Analítica: procesar datos, comparar, calcular, revisar información
- Cognitiva: decidir, aprobar, planificar, diseñar, resolver excepciones

IMPORTANTE: Devuelve ÚNICAMENTE un objeto JSON con datos reales del proceso.
NO devuelvas un schema JSON. NO devuelvas definiciones de tipos ni estructuras
con claves como '$defs', 'properties', 'required', 'title' o 'type' en el nivel raíz.
""".strip()

HUMAN_PROMPT = """
Analiza el siguiente proceso y extrae su representación estructurada AS-IS.

DESCRIPCIÓN DEL PROCESO:
{raw_input}

FORMATO DE SALIDA — devuelve SOLO este objeto JSON con datos reales:

{{
  "id": "PROC-001",
  "name": "Nombre real del proceso",
  "description": "Descripción real del proceso",
  "owner": "Área o rol responsable del proceso",
  "scope": "Desde [inicio] hasta [fin del proceso]",
  "participants": ["Rol 1", "Rol 2"],
  "systems": ["Sistema A", "Sistema B"],
  "activities": [
    {{
      "id": "ACT-001",
      "name": "Nombre de la actividad",
      "description": "Qué hace exactamente esta actividad",
      "responsible": "Rol que la ejecuta",
      "type": "operativa",
      "estimated_duration_min": 15,
      "depends_on": [],
      "systems_used": []
    }},
    {{
      "id": "ACT-002",
      "name": "Segunda actividad",
      "description": "Descripción de la segunda actividad",
      "responsible": "Otro rol",
      "type": "analitica",
      "estimated_duration_min": 30,
      "depends_on": ["ACT-001"],
      "systems_used": ["ERP"]
    }}
  ]
}}

EJEMPLO CONCRETO — Proceso de Facturación:

{{
  "id": "PROC-001",
  "name": "Proceso de Facturación",
  "description": "Generación y envío de facturas a clientes",
  "owner": "Área de Finanzas",
  "scope": "Desde recepción de pedido hasta confirmación de envío de factura",
  "participants": ["Asistente de finanzas", "Jefe de finanzas"],
  "systems": ["CRM", "SAP"],
  "activities": [
    {{
      "id": "ACT-001",
      "name": "Descargar pedido del CRM",
      "description": "El asistente descarga el pedido manualmente del CRM",
      "responsible": "Asistente de finanzas",
      "type": "operativa",
      "estimated_duration_min": 15,
      "depends_on": [],
      "systems_used": ["CRM"]
    }},
    {{
      "id": "ACT-002",
      "name": "Generar factura en SAP",
      "description": "Se genera la factura con los datos del pedido en SAP",
      "responsible": "Asistente de finanzas",
      "type": "operativa",
      "estimated_duration_min": 20,
      "depends_on": ["ACT-001"],
      "systems_used": ["SAP"]
    }},
    {{
      "id": "ACT-003",
      "name": "Revisar y aprobar factura",
      "description": "El jefe revisa la factura antes de enviarla al cliente",
      "responsible": "Jefe de finanzas",
      "type": "cognitiva",
      "estimated_duration_min": 10,
      "depends_on": ["ACT-002"],
      "systems_used": ["SAP"]
    }}
  ]
}}

Usa el proceso descrito arriba para generar un JSON equivalente con datos reales.
Sin markdown, sin bloques de código, solo el objeto JSON puro.
""".strip()

extract_asis_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])
