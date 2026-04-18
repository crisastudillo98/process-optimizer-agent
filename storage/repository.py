from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from storage.models import Analysis

def create_analysis(db: Session, session_id: str, process_name: str, raw_input: str) -> Analysis:
    """Registra un análisis nuevo al iniciar el pipeline."""
    record = Analysis(
        id=session_id,
        process_name=process_name or "Sin nombre",
        raw_input=raw_input,
        status="running",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def complete_analysis(db: Session, session_id: str, result: dict, score: float = None) -> Optional[Analysis]:
    """Marca el análisis como completado y guarda el resultado."""
    record = db.query(Analysis).filter(Analysis.id == session_id).first()
    if not record:
        return None

    record.status       = "completed"
    record.result_json  = json.dumps(result, default=str)
    record.completed_at = datetime.now(timezone.utc)
    record.score        = score

    # Extraer métricas rápidas si están disponibles
    kpi = result.get("kpi_report") or {}
    if kpi:
        ct = kpi.get("cycle_time", {})
        ac = kpi.get("automation_coverage", {})
        record.cycle_time_reduction_pct = ct.get("reduction_pct")
        record.automation_coverage_pct  = ac.get("tobe_value")

    db.commit()
    db.refresh(record)
    return record

def fail_analysis(db: Session, session_id: str, errors: list[str]) -> Optional[Analysis]:
    """Marca el análisis como fallido."""
    record = db.query(Analysis).filter(Analysis.id == session_id).first()
    if not record:
        return None
    record.status     = "error"
    record.has_errors = True
    record.result_json = json.dumps({"errors": errors})
    record.completed_at = datetime.now(timezone.utc)
    db.commit()
    return record

def get_analysis(db: Session, session_id: str) -> Optional[Analysis]:
    return db.query(Analysis).filter(Analysis.id == session_id).first()

def list_analyses(db: Session, limit: int = 50, offset: int = 0) -> list[Analysis]:
    """Lista análisis ordenados por fecha descendente."""
    return (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
