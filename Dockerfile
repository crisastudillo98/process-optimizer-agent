# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Dependencias del sistema para lxml y PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Pre-descargar modelo FastEmbed durante el build ───────
# Se descarga aquí donde hay permisos de root
#RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Solo las libs del sistema necesarias en runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes instalados desde el builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Copiar cache del modelo FastEmbed desde el builder ────
#COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copiar código fuente
COPY . .

# Crear directorios de trabajo necesarios
RUN mkdir -p \
    storage/outputs/bpmn \
    storage/outputs/reports \
    storage/vector_db \
    /tmp/process_optimizer

# Usuario no-root para producción
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser
USER appuser

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser \
    LOG_LEVEL=INFO \
    HITL_ENABLED=true \
    OPENAI_MODEL=gpt-4o \
    OPENAI_TEMPERATURE=0.2 \
    OPENAI_TEMPERATURE=0.2 
    #HF_HOME=/root/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]