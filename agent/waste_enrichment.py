"""
Sprint 8 — Muda / Mura / Muri classification + tool recommendations.

Pure functions over the existing schemas. No LLM calls — entirely derived from
WasteAnalysisResult, Process, and TOBEProcess so the classification is fast
and reproducible.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Optional

from models.schemas import (
    Process,
    TOBEProcess,
    WasteAnalysisResult,
    WasteType,
    ActivityType,
)


# ─────────────────────────────────────────────
# Muda / Mura / Muri
# ─────────────────────────────────────────────

# Heuristic — Mura threshold: activities whose duration is more than
# this many standard deviations away from the mean count as "variable".
_MURA_SIGMA = 1.0

# Muri threshold: an activity whose duration exceeds the team's average
# activity duration by this multiplier (and is owned by a single role)
# is flagged as overburden for that role.
_MURI_DURATION_MULT = 2.0


def classify_muda_mura_muri(
    asis: Process,
    waste: WasteAnalysisResult,
) -> dict:
    """
    Returns a dict with three lists:
      - muda: activities classified as waste (non-value-adding).
      - mura: activities whose duration is significantly different from peers (variability).
      - muri: roles overloaded with a long, complex activity (overburden).
    """
    muda: list[dict] = []
    mura: list[dict] = []
    muri: list[dict] = []

    # Muda — drawn from the waste analysis (already classified by the LLM).
    for d in waste.activity_details or []:
        if d.waste_classification and d.waste_classification.value == "desperdicio":
            muda.append({
                "activity_id":   d.activity_id,
                "activity_name": d.activity_name,
                "waste_type":    d.waste_type.value if d.waste_type else None,
                "justification": d.waste_justification or "",
                "waste_time_min": round(d.estimated_waste_time_min or 0.0, 1),
            })

    # Mura — duration variability across activities.
    durations = [
        (a.id, a.name, a.estimated_duration_min, a.responsible)
        for a in asis.activities
        if a.estimated_duration_min is not None and a.estimated_duration_min > 0
    ]
    if len(durations) >= 3:
        vals = [d[2] for d in durations]
        avg = mean(vals)
        sd  = pstdev(vals) or 0.0
        if sd > 0:
            for aid, name, dur, owner in durations:
                if abs(dur - avg) >= _MURA_SIGMA * sd:
                    mura.append({
                        "activity_id":   aid,
                        "activity_name": name,
                        "duration_min":  round(dur, 1),
                        "avg_duration_min": round(avg, 1),
                        "deviation_min": round(dur - avg, 1),
                        "responsible":   owner,
                    })

    # Muri — overburden detection: long activities concentrated on one role.
    if asis.activities:
        avg_dur = mean(
            [a.estimated_duration_min or 0.0 for a in asis.activities]
        ) or 1.0
        load_by_role: dict[str, float] = {}
        for a in asis.activities:
            if a.responsible:
                load_by_role[a.responsible] = load_by_role.get(a.responsible, 0.0) + (a.estimated_duration_min or 0.0)
        team_avg_load = mean(load_by_role.values()) if load_by_role else 0.0

        for a in asis.activities:
            dur = a.estimated_duration_min or 0.0
            if dur >= avg_dur * _MURI_DURATION_MULT and a.responsible:
                role_load = load_by_role.get(a.responsible, 0.0)
                muri.append({
                    "activity_id":   a.id,
                    "activity_name": a.name,
                    "responsible":   a.responsible,
                    "duration_min":  round(dur, 1),
                    "role_total_load_min": round(role_load, 1),
                    "team_avg_load_min":   round(team_avg_load, 1),
                    "reason": (
                        f"La actividad dura {dur:.0f} min ({dur/avg_dur:.1f}× el promedio del proceso) "
                        f"y recae en {a.responsible}."
                    ),
                })

    return {"muda": muda, "mura": mura, "muri": muri}


# ─────────────────────────────────────────────
# Tool recommendations
# ─────────────────────────────────────────────

# Catalog organized by "tag" — derived from waste types, activity type, and
# automatable flags. Each entry lists candidate tools, an order-of-magnitude
# cost range, expected ROI horizon, and the methodology it embodies.
TOOL_CATALOG: dict[str, dict] = {
    "document_processing": {
        "trigger_waste": [WasteType.OVERPROCESSING.value, WasteType.MOTION.value, WasteType.DEFECTS.value],
        "trigger_keywords": ["factura", "factur", "document", "captura", "ingreso", "manual"],
        "tools": ["UiPath", "Power Automate", "Automation Anywhere"],
        "cost_range": "$500–2000/mes",
        "roi_months": 6,
        "methodology": "Lean — Eliminación de Muda de movimiento y sobreproceso",
    },
    "approval_workflow": {
        "trigger_waste": [WasteType.WAITING.value, WasteType.OVERPROCESSING.value],
        "trigger_keywords": ["aprob", "revis", "valid", "firma"],
        "tools": ["ServiceNow", "Monday.com", "Jira Service Desk"],
        "cost_range": "$300–1500/mes",
        "roi_months": 4,
        "methodology": "Six Sigma — Reducción de variabilidad en aprobaciones",
    },
    "analytics_dashboards": {
        "trigger_waste": [WasteType.UNUSED_TALENT.value, WasteType.OVERPROCESSING.value],
        "trigger_keywords": ["report", "reporte", "métric", "metric", "dashboard", "kpi"],
        "tools": ["Power BI", "Tableau", "Looker Studio"],
        "cost_range": "$10–70/usuario/mes",
        "roi_months": 5,
        "methodology": "Kaizen — Mejora continua basada en datos",
    },
    "communication_orchestration": {
        "trigger_waste": [WasteType.WAITING.value, WasteType.TRANSPORT.value],
        "trigger_keywords": ["correo", "email", "outlook", "comunicac", "notific"],
        "tools": ["Slack", "Microsoft Teams + Power Automate"],
        "cost_range": "$8–25/usuario/mes",
        "roi_months": 3,
        "methodology": "Lean — Reducción de Muda de transporte de información",
    },
    "data_capture_forms": {
        "trigger_waste": [WasteType.DEFECTS.value, WasteType.OVERPROCESSING.value],
        "trigger_keywords": ["form", "formulario", "encuesta", "captur"],
        "tools": ["Microsoft Forms", "Jotform", "Typeform"],
        "cost_range": "$15–80/mes",
        "roi_months": 2,
        "methodology": "Six Sigma — Reducción de defectos por error humano",
    },
    "rpa_attended": {
        "trigger_waste": [WasteType.MOTION.value, WasteType.OVERPROCESSING.value],
        "trigger_keywords": ["copiar", "pegar", "trasla", "cargar", "exportar"],
        "tools": ["UiPath StudioX", "Power Automate Desktop"],
        "cost_range": "$25–150/bot/mes",
        "roi_months": 4,
        "methodology": "Lean — Automatización de Muda de movimiento",
    },
    "queue_optimization": {
        "trigger_waste": [WasteType.WAITING.value, WasteType.INVENTORY.value],
        "trigger_keywords": ["cola", "espera", "pendiente", "backlog"],
        "tools": ["Kanbanize", "Trello + Butler", "Asana Rules"],
        "cost_range": "$10–25/usuario/mes",
        "roi_months": 3,
        "methodology": "Kaizen + Lean — Visibilidad y reducción de Muda de espera",
    },
}


def _tag_for_activity(act_name: str, waste_type: Optional[str], description: str = "") -> Optional[str]:
    """Match an activity to a tool-recommendation tag by waste type + keyword overlap."""
    text = f"{act_name} {description}".lower()
    # Priority 1 — exact waste_type match.
    for tag, entry in TOOL_CATALOG.items():
        if waste_type and waste_type in entry["trigger_waste"]:
            if any(kw in text for kw in entry["trigger_keywords"]):
                return tag
    # Priority 2 — keyword-only fallback.
    for tag, entry in TOOL_CATALOG.items():
        if any(kw in text for kw in entry["trigger_keywords"]):
            return tag
    return None


def recommend_tools(
    tobe: TOBEProcess,
    waste: WasteAnalysisResult,
) -> list[dict]:
    """
    For each TO-BE activity that is automated/optimized, suggest a concrete
    tool from the catalog. Falls back to the activity's own automation_tool
    if no catalog tag matches.
    """
    waste_by_activity: dict[str, str] = {}
    for d in waste.activity_details or []:
        if d.waste_type:
            waste_by_activity[d.activity_name.lower()] = d.waste_type.value

    recs: list[dict] = []
    for a in tobe.activities:
        status = a.status.value if hasattr(a.status, "value") else str(a.status)
        if status not in ("automatizada", "optimizada", "combinada"):
            continue
        wt = waste_by_activity.get(a.name.lower())
        tag = _tag_for_activity(a.name, wt, a.description or "")
        entry = TOOL_CATALOG.get(tag) if tag else None
        if entry:
            recs.append({
                "activity_id":   a.id,
                "activity_name": a.name,
                "category":      tag,
                "tools":         entry["tools"],
                "cost_range":    entry["cost_range"],
                "roi_months":    entry["roi_months"],
                "methodology":   entry["methodology"],
                "fallback_tool": a.automation_tool,
            })
        elif a.automation_tool:
            recs.append({
                "activity_id":   a.id,
                "activity_name": a.name,
                "category":      "custom",
                "tools":         [a.automation_tool],
                "cost_range":    "—",
                "roi_months":    None,
                "methodology":   "Definido por la propuesta TO-BE",
                "fallback_tool": a.automation_tool,
            })
    return recs


def build_enrichment_block(
    asis: Process,
    tobe: TOBEProcess,
    waste: WasteAnalysisResult,
) -> dict:
    """One-stop helper used by the final report builder."""
    classification = classify_muda_mura_muri(asis, waste)
    tools = recommend_tools(tobe, waste)
    return {
        "waste_classification": classification,
        "tool_recommendations": tools,
        "summary": {
            "muda_count": len(classification["muda"]),
            "mura_count": len(classification["mura"]),
            "muri_count": len(classification["muri"]),
            "tool_recommendation_count": len(tools),
        },
    }
