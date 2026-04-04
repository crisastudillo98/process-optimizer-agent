.PHONY: install dev test test-integration build up down seed logs clean

# ── Desarrollo ────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --ignore=tests/test_api.py -m "not integration"

test-api:
	pytest tests/test_api.py -v

test-all:
	pytest tests/ -v -m "not integration"

test-integration:
	pytest tests/ -v -m integration

coverage:
	pytest tests/ --cov=agent --cov=api --cov=rag \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    -m "not integration"

# ── Docker ────────────────────────────────────────────────────────────────────
build:
	docker compose build --no-cache

up:
	docker compose up -d

down:
	docker compose down

seed:
	docker compose run --rm seed-knowledge

logs:
	docker compose logs -f process-optimizer

restart:
	docker compose restart process-optimizer

# ── Limpieza ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf .pytest_cache htmlcov .coverage

clean-storage:
	rm -rf storage/outputs/bpmn/* storage/outputs/reports/*
	@echo "⚠️  Vector DB NO fue eliminada. Usa clean-all para eliminarla también."

clean-all: clean clean-storage
	rm -rf storage/vector_db/*
	@echo "✅ Todo limpiado incluyendo Vector DB."