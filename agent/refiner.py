from __future__ import annotations
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models.schemas import TOBEProcess, OptimizedActivity, ActivityStatus
from llm.factory import get_llm
from observability.logger import get_logger

logger = get_logger(__name__)


def _strip_markdown_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()
    return text.strip()


_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a process optimization expert specializing in Lean, Six Sigma, and Kaizen.
You receive a TO-BE process in JSON and a natural-language instruction to modify it.

Rules:
- Keep all fields unrelated to the change unchanged
- Valid ActivityStatus values: conservada, optimizada, automatizada, eliminada, combinada, nueva
- Valid ActivityType values: operativa, analitica, cognitiva
- total_duration_min = sum of estimated_duration_min for non-eliminated activities
- Return ONLY valid JSON — no markdown, no explanations, no code blocks"""),
    ("human", """Current TO-BE process:
{current_tobe}

Instruction: {instruction}

Return the modified process as JSON with the same structure."""),
])


def refine_tobe_process(current_tobe: TOBEProcess, instruction: str) -> TOBEProcess:
    """
    Calls the LLM to modify the TO-BE process according to a natural-language instruction.
    Returns a fully validated TOBEProcess.
    """
    llm = get_llm(temperature=0.3)
    chain = _REFINE_PROMPT | llm | StrOutputParser()

    raw = chain.invoke({
        "current_tobe": current_tobe.model_dump_json(indent=2),
        "instruction":  instruction,
    })

    data = json.loads(_strip_markdown_json(raw))

    activities: list[OptimizedActivity] = []
    for act in data.get("activities", []):
        act["status"] = act.get("status", "conservada").lower()
        act["type"]   = act.get("type",   "operativa").lower()
        if act["status"] == ActivityStatus.ELIMINATED:
            act["estimated_duration_min"] = 0.0
            act["duration_reduction_pct"] = 100.0
        activities.append(OptimizedActivity(**act))

    total_duration = sum(
        (a.estimated_duration_min or 0.0) for a in activities
        if a.status != ActivityStatus.ELIMINATED
    )

    return TOBEProcess(
        id=data.get("id", current_tobe.id),
        original_process_id=data.get("original_process_id", current_tobe.original_process_id),
        name=data.get("name", current_tobe.name),
        description=data.get("description", current_tobe.description),
        owner=data.get("owner", current_tobe.owner),
        activities=activities,
        total_duration_min=total_duration,
        applied_methodologies=data.get("applied_methodologies", current_tobe.applied_methodologies),
        sipoc=data.get("sipoc", current_tobe.sipoc),
        human_approved=True,
    )
