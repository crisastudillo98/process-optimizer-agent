from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from storage.models import Analysis


def create_analysis(
    db: Session,
    session_id: str,
    process_name: str,
    raw_input: str,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Analysis:
    record = Analysis(
        id=session_id,
        process_name=process_name or "Sin nombre",
        raw_input=raw_input,
        status="running",
        user_id=user_id,
        tenant_id=tenant_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def complete_analysis(
    db: Session,
    session_id: str,
    result: dict,
    score: float = None,
) -> Optional[Analysis]:
    record = db.query(Analysis).filter(Analysis.id == session_id).first()
    if not record:
        return None

    record.status       = "completed"
    record.result_json  = json.dumps(result, default=str)
    record.completed_at = datetime.now(timezone.utc)
    record.score        = score

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
    record = db.query(Analysis).filter(Analysis.id == session_id).first()
    if not record:
        return None
    record.status      = "error"
    record.has_errors  = True
    record.result_json = json.dumps({"errors": errors})
    record.completed_at = datetime.now(timezone.utc)
    db.commit()
    return record


def get_analysis(
    db: Session,
    session_id: str,
    user_id: Optional[str] = None,
) -> Optional[Analysis]:
    record = db.query(Analysis).filter(Analysis.id == session_id).first()
    if not record:
        return None
    # Ownership check: rows without user_id are pre-auth legacy rows, always accessible
    if user_id and record.user_id and record.user_id != user_id:
        return None
    return record


def list_analyses(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    user_id: Optional[str] = None,
) -> list[Analysis]:
    q = db.query(Analysis)
    if user_id:
        # Return rows belonging to this user OR legacy rows (no user_id)
        from sqlalchemy import or_
        q = q.filter(or_(Analysis.user_id == user_id, Analysis.user_id == None))
    return (
        q.order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
