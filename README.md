# 🔄 Process Optimizer Agent

Agente inteligente para la **optimización automática de procesos empresariales**.
Transforma descripciones AS-IS en lenguaje natural en propuestas TO-BE optimizadas
con **Lean, Six Sigma y Kaizen**, generando diagramas BPMN 2.0 y KPIs cuantitativos.

---

## 🏗️ Arquitectura

```
raw_input (texto / PDF / Excel)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                  │
│                                                         │
│  load_document → extract_asis → analyze_waste           │
│       → retrieve_rag → optimize_tobe                    │
│       → [HITL review] → generate_bpmn → calculate_kpis  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
  BPMN XML 2.0 + KPI Report + TO-BE estructurado
```

### Stack tecnológico

| Capa | Tecnología |
|---|---|
| LLM (default) | **Llama 3.3 70B Versatile** via [Groq](https://groq.com) |
| LLM (alternativo) | GPT-4o via OpenAI |
| Orquestación | LangGraph |
| RAG | ChromaDB + `DefaultEmbeddingFunction` (local, sin costo) |
| Document Loader | PyMuPDF + openpyxl |
| BPMN | lxml + XML BPMN 2.0 |
| Schemas | Pydantic v2 |
| API | FastAPI |
| Contenedor | Docker + Docker Compose |

> **Nota sobre embeddings:** el agente usa `DefaultEmbeddingFunction` de ChromaDB
> (`all-MiniLM-L6-v2` local). No requiere API key de OpenAI para los embeddings.

---

## 🚀 Inicio rápido

### 1. Clonar y configurar

```bash
git clone https://github.com/tu-usuario/process-optimizer-agent
cd process-optimizer-agent
cp .env.example .env
# Editar .env y agregar GROQ_API_KEY (o OPENAI_API_KEY si usas OpenAI)
```

### 2. Con Docker (recomendado)

```bash
# Build y levantar todos los servicios
make build
make up

# Inicializar la knowledge base Lean/Six Sigma (solo primera vez)
make seed

# Ver logs
make logs
```

### 3. Desarrollo local

```bash
# Instalar dependencias
make install

# Inicializar knowledge base
python -m rag.seed_knowledge

# Levantar en modo desarrollo con hot-reload
make dev
```

La API estará disponible en:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🤖 Proveedores LLM soportados

El agente soporta múltiples proveedores configurables via `LLM_PROVIDER`:

| Provider | Variable | Modelo default | Estado |
|---|---|---|---|
| `groq` ✅ | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | **Default — recomendado** |
| `openai` | `OPENAI_API_KEY` | `gpt-4o` | Soportado |
| `perplexity` | — | — | Experimental |

Para cambiar de proveedor basta con editar el `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## 📖 Uso de la API

### Flujo completo

```bash
# 1. Analizar un proceso desde texto
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "El proceso de facturación inicia cuando el área comercial
    notifica al equipo de finanzas sobre un pedido aprobado. Un asistente
    descarga el pedido del CRM (15 min), verifica datos en Excel (20 min),
    genera la factura en SAP (25 min) y espera confirmación 3 días hábiles."
  }'

# Respuesta:
# { "session_id": "abc-123", "status": "running", ... }


# 2. Consultar progreso (polling)
curl http://localhost:8000/sessions/abc-123/status


# 3. Obtener reporte completo cuando kpi_ok=true
curl http://localhost:8000/sessions/abc-123/report


# 4. Descargar BPMN
curl http://localhost:8000/sessions/abc-123/bpmn \
  -o proceso_optimizado.bpmn
```

### Analizar desde archivo

```bash
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@mi_proceso.pdf"
```

### Revisión humana (HITL)

```bash
# Aprobar el TO-BE generado
curl -X POST http://localhost:8000/sessions/abc-123/review \
  -H "Content-Type: application/json" \
  -d '{ "approved": true, "feedback": "Excelente propuesta." }'

# Rechazar y pedir re-optimización con feedback
curl -X POST http://localhost:8000/sessions/abc-123/review \
  -H "Content-Type: application/json" \
  -d '{
    "approved": false,
    "feedback": "El paso de aprobación no puede ser automatizado por política interna.
                 Mantenlo como tarea manual pero optimiza el tiempo de respuesta."
  }'
```

---

## 🧩 Componentes del agente

| Nodo | Archivo | Función |
|---|---|---|
| `load_document` | `agent/document_loader.py` | Carga PDF, Excel, JSON, texto libre |
| `extract_asis` | `agent/process_extractor.py` | LLM extrae proceso AS-IS estructurado |
| `analyze_waste` | `agent/analyzer.py` | Detecta Muda, redundancias, O/A/C |
| `retrieve_rag` | `rag/retriever.py` | Recupera casos similares + patrones Lean |
| `optimize_tobe` | `agent/optimizer.py` | Genera propuesta TO-BE optimizada |
| `hitl_review` | `agent/optimizer.py` | Pausa para validación humana |
| `generate_bpmn` | `agent/bpmn_generator.py` | XML BPMN 2.0 determinístico |
| `calculate_kpis` | `agent/kpi_calculator.py` | 5 KPIs + ROI + Sigma + enriquecimiento LLM |

---

## 📊 KPIs generados

| KPI | Descripción | Cálculo |
|---|---|---|
| Tiempo de ciclo | Reducción AS-IS vs TO-BE | Determinístico |
| Actividades manuales | Liberación de carga manual | Determinístico |
| Muda eliminada | % desperdicio Lean removido | Determinístico + ponderado |
| Cobertura automatización | % actividades con `status=automatizada` | Determinístico |
| Eficiencia (VAR) | Process Time / Lead Time | Determinístico |
| ROI estimado | Ahorro anual / costo implementación | Determinístico |
| Nivel Sigma | 2σ → 6σ según % desperdicio | Lookup table |
| Interpretaciones | Contexto negocio + benchmarks | LLM (no bloqueante) |

---

## 🧪 Tests

```bash
# Tests unitarios (sin API real)
make test

# Tests de la API FastAPI
make test-api

# Todos los tests unitarios
make test-all

# Tests de integración (requieren GROQ_API_KEY + ChromaDB)
make test-integration

# Coverage report
make coverage
```

### Estructura de tests

```
tests/
├── conftest.py                    # Fixtures globales
├── test_schemas.py                # Validación de modelos Pydantic
├── test_document_loader.py        # Carga de archivos
├── test_extractor.py              # Extracción AS-IS
├── test_analyzer.py               # Análisis Lean
├── test_optimizer.py              # Generación TO-BE
├── test_bpmn_generator.py         # Generación BPMN
├── test_kpi_calculator.py         # Cálculo de KPIs
├── test_rag.py                    # Pipeline RAG
├── test_api.py                    # Endpoints FastAPI
└── test_integration_pipeline.py   # E2E sin mocks
```

---

## 🏗️ Estructura del proyecto

```
process-optimizer-agent/
├── agent/                  # Nodos del grafo LangGraph
│   ├── orchestrator.py     # StateGraph — cerebro del agente
│   ├── document_loader.py  # PDF, Excel, texto libre
│   ├── process_extractor.py
│   ├── analyzer.py         # Muda, redundancias, O/A/C
│   ├── optimizer.py        # TO-BE + HITL
│   ├── bpmn_generator.py   # XML BPMN 2.0
│   └── kpi_calculator.py
├── prompts/                # Prompts versionados por nodo
├── rag/                    # Pipeline RAG
│   ├── embedder.py         # DefaultEmbeddingFunction (local)
│   ├── vector_store.py     # ChromaDB
│   ├── retriever.py
│   └── seed_knowledge.py   # KB Lean/Six Sigma
├── models/
│   └── schemas.py          # 20+ modelos Pydantic v2
├── config/
│   └── settings.py         # Pydantic BaseSettings
├── observability/
│   ├── tracer.py
│   └── logger.py
├── api/
│   └── main.py             # 16 endpoints FastAPI
├── tests/
├── storage/
│   ├── outputs/bpmn/
│   ├── outputs/reports/
│   └── vector_db/
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## ⚙️ Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `LLM_PROVIDER` | Proveedor LLM (`groq`, `openai`, `perplexity`) | `groq` |
| `GROQ_API_KEY` | API key de Groq | **Requerida** (si `LLM_PROVIDER=groq`) |
| `GROQ_MODEL` | Modelo Groq | `llama-3.3-70b-versatile` |
| `OPENAI_API_KEY` | API key de OpenAI | Requerida si `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | Modelo OpenAI | `gpt-4o` |
| `EMBEDDING_MODEL` | Modelo embeddings (`local` o nombre OpenAI) | `local` |
| `HITL_ENABLED` | Activa revisión humana | `true` |
| `RAG_TOP_K` | Resultados del retrieval | `5` |
| `LOG_LEVEL` | Nivel de logging | `INFO` |
| `LANGSMITH_API_KEY` | Trazabilidad LangSmith | Opcional |

---

## ⚠️ Limitaciones conocidas

- **HITL sin timeout:** las sesiones en espera de revisión humana no expiran automáticamente. Si no se llama a `/review`, el pipeline queda pausado indefinidamente.
- **VAR siempre 100%:** el Value-Added Ratio puede reportarse como 100% cuando todas las actividades del AS-IS tienen valor agregado declarado. No refleja esperas implícitas no modeladas.
- **ROI sensible al input:** el ROI estimado depende de los tiempos declarados en el texto de entrada. Entradas muy cortas o ambiguas generan estimaciones poco realistas.
- **`perplexity` como provider:** está disponible en el enum pero no tiene configuración activa. No usar en producción.

---

## 🔭 Roadmap

- [ ] Soporte multiidioma en extracción (inglés, portugués)
- [ ] Exportación a PDF del reporte completo
- [ ] Interfaz web Streamlit para usuarios no técnicos
- [ ] Integración con n8n para despliegue automático de flujos RPA
- [ ] Soporte process mining desde logs CSV/XES con PM4Py
- [ ] Autenticación OAuth2 en la API
- [ ] Timeout configurable para sesiones HITL
- [ ] Soporte Perplexity API como provider LLM

---

## 📄 Licencia

MIT — Libre para uso académico y comercial.

## 👥 Autores

Desarrollado como extensión del proyecto MIAA — Universidad Icesi.
