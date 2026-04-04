from __future__ import annotations
import json
import uuid
from pathlib import Path
from datetime import datetime

from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential
#from langchain_openai import ChatOpenAI
#from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
from models.schemas import (
    AgentState,
    TOBEProcess,
    ActivityStatus,
    ActivityType,
    BPMNStructure,
    BPMNElement,
    BPMNElementType,
    BPMNSequenceFlow,
    BPMNLane,
    BPMNOutput,
)
#from prompts.generate_bpmn import generate_bpmn_prompt
from observability.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# NAMESPACES BPMN 2.0
# ─────────────────────────────────────────────

BPMN_NS   = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
DC_NS     = "http://www.omg.org/spec/DD/20100524/DC"
DI_NS     = "http://www.omg.org/spec/DD/20100524/DI"

NSMAP = {
    None:     BPMN_NS,
    "bpmndi": BPMNDI_NS,
    "dc":     DC_NS,
    "di":     DI_NS,
}


# ─────────────────────────────────────────────
# CONSTRUCCIÓN DE LA ESTRUCTURA BPMN DESDE TO-BE
# (sin LLM — determinístico)
# ─────────────────────────────────────────────

def _map_activity_to_bpmn_type(activity) -> BPMNElementType:
    """
    Mapea una OptimizedActivity a su tipo BPMN correspondiente.
    Reglas: automatizada → serviceTask | cognitiva → userTask | resto → task
    """
    if activity.status == ActivityStatus.AUTOMATED:
        return BPMNElementType.SERVICE_TASK
    if activity.type == ActivityType.COGNITIVE:
        return BPMNElementType.USER_TASK
    return BPMNElementType.TASK


def _needs_gateway(activity_name: str) -> bool:
    """Detecta si una actividad implica una decisión (gateway)."""
    decision_keywords = {
        "aprobación", "aprueba", "decisión", "decide",
        "validación", "valida", "autorización", "autoriza",
        "verificación", "verifica",
    }
    name_lower = activity_name.lower()
    return any(kw in name_lower for kw in decision_keywords)


def build_bpmn_structure_from_tobe(tobe: TOBEProcess) -> BPMNStructure:
    """
    Construye la estructura BPMN de forma determinística desde el TO-BE.
    No usa LLM — garantiza que cada actividad válida esté representada.
    """
    elements: list[BPMNElement]           = []
    flows: list[BPMNSequenceFlow]         = []
    lanes_map: dict[str, BPMNLane]        = {}
    flow_counter    = 1
    gateway_counter = 1

    # ── StartEvent ────────────────────────────────────────────────────────
    start_id = "start_1"
    elements.append(BPMNElement(
        id=start_id,
        type=BPMNElementType.START_EVENT,
        name="Inicio",
        documentation="Inicio del proceso optimizado TO-BE",
    ))

    previous_id = start_id

    # ── Actividades (excluye eliminadas) ──────────────────────────────────
    valid_activities = [
        a for a in tobe.activities
        if a.status != ActivityStatus.ELIMINATED
    ]

    for idx, activity in enumerate(valid_activities, start=1):
        task_id   = f"task_{idx}"
        bpmn_type = _map_activity_to_bpmn_type(activity)

        # Registrar lane (carril por responsable)
        responsible = activity.responsible or "Sin asignar"
        if responsible not in lanes_map:
            lane_id = f"lane_{len(lanes_map) + 1}"
            lanes_map[responsible] = BPMNLane(
                id=lane_id,
                name=responsible,
                element_ids=[],
            )
        lanes_map[responsible].element_ids.append(task_id)

        # Construir documentación enriquecida
        doc_parts = [activity.description]
        if activity.automation_tool:
            doc_parts.append(f"Herramienta: {activity.automation_tool}")
        if activity.improvement_justification:
            doc_parts.append(f"Mejora: {activity.improvement_justification}")
        doc_parts.append(
            f"Reducción de tiempo: {activity.duration_reduction_pct:.0f}%"
        )

        elements.append(BPMNElement(
            id=task_id,
            type=bpmn_type,
            name=activity.name[:50],
            lane=responsible,
            documentation=" | ".join(doc_parts),
        ))

        # Flujo desde el elemento anterior a esta tarea
        flows.append(BPMNSequenceFlow(
            id=f"flow_{flow_counter}",
            source_ref=previous_id,
            target_ref=task_id,
        ))
        flow_counter += 1

        # ── Gateway de decisión (si aplica) ───────────────────────────────
        if _needs_gateway(activity.name) and idx < len(valid_activities):
            gw_id = f"gateway_{gateway_counter}"
            gateway_counter += 1

            elements.append(BPMNElement(
                id=gw_id,
                type=BPMNElementType.EXCLUSIVE_GATEWAY,
                name="¿Aprobado?",
                lane=responsible,
            ))

            # Flujo tarea → gateway
            flows.append(BPMNSequenceFlow(
                id=f"flow_{flow_counter}",
                source_ref=task_id,
                target_ref=gw_id,
            ))
            flow_counter += 1

            # Flujo gateway → siguiente (aprobado)
            next_id = (
                f"task_{idx + 1}" if idx < len(valid_activities)
                else "end_1"
            )
            flows.append(BPMNSequenceFlow(
                id=f"flow_{flow_counter}",
                source_ref=gw_id,
                target_ref=next_id,
                name="Aprobado",
            ))
            flow_counter += 1

            # Flujo gateway → tarea anterior (rechazado = reproceso)
            flows.append(BPMNSequenceFlow(
                id=f"flow_{flow_counter}",
                source_ref=gw_id,
                target_ref=task_id,
                name="Rechazado",
            ))
            flow_counter += 1

            previous_id = gw_id
        else:
            previous_id = task_id

    # ── EndEvent ──────────────────────────────────────────────────────────
    end_id = "end_1"
    elements.append(BPMNElement(
        id=end_id,
        type=BPMNElementType.END_EVENT,
        name="Fin",
        documentation="Fin del proceso optimizado TO-BE",
    ))

    flows.append(BPMNSequenceFlow(
        id=f"flow_{flow_counter}",
        source_ref=previous_id,
        target_ref=end_id,
    ))

    logger.info(
        f"Estructura BPMN construida: {len(elements)} elementos, "
        f"{len(flows)} flujos, {len(lanes_map)} carriles"
    )

    return BPMNStructure(
        process_id=tobe.id,
        process_name=tobe.name,
        lanes=list(lanes_map.values()),
        elements=elements,
        sequence_flows=flows,
    )


# ─────────────────────────────────────────────
# GENERACIÓN XML BPMN 2.0 con lxml
# ─────────────────────────────────────────────

def _make_tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _build_xml_from_structure(structure: BPMNStructure) -> str:
    """
    Convierte BPMNStructure a XML BPMN 2.0 válido usando lxml.
    Compatible con Camunda, Bizagi y cualquier herramienta BPMN 2.0.
    """
    # ── Raíz: definitions ─────────────────────────────────────────────────
    root = etree.Element(
        _make_tag(BPMN_NS, "definitions"),
        nsmap=NSMAP,
        attrib={
            "id":              f"Definitions_{uuid.uuid4().hex[:8]}",
            "targetNamespace": "http://process-optimizer-agent",
            "exporter":        "Process Optimizer Agent",
            "exporterVersion": "1.0",
        }
    )

    # ── Collaboration (pool) ──────────────────────────────────────────────
    collaboration = etree.SubElement(
        root,
        _make_tag(BPMN_NS, "collaboration"),
        attrib={"id": "Collaboration_1"},
    )
    participant = etree.SubElement(
        collaboration,
        _make_tag(BPMN_NS, "participant"),
        attrib={
            "id":         "Participant_1",
            "name":       structure.process_name[:60],
            "processRef": f"Process_{structure.process_id}",
        },
    )

    # ── Process ───────────────────────────────────────────────────────────
    process_el = etree.SubElement(
        root,
        _make_tag(BPMN_NS, "process"),
        attrib={
            "id":          f"Process_{structure.process_id}",
            "name":        structure.process_name[:60],
            "isExecutable": "false",
        },
    )

    # ── LaneSet (carriles por responsable) ────────────────────────────────
    if structure.lanes:
        lane_set = etree.SubElement(
            process_el,
            _make_tag(BPMN_NS, "laneSet"),
            attrib={"id": "LaneSet_1"},
        )
        for lane in structure.lanes:
            lane_el = etree.SubElement(
                lane_set,
                _make_tag(BPMN_NS, "lane"),
                attrib={"id": lane.id, "name": lane.name},
            )
            for elem_id in lane.element_ids:
                ref_el = etree.SubElement(
                    lane_el,
                    _make_tag(BPMN_NS, "flowNodeRef"),
                )
                ref_el.text = elem_id

    # ── Elementos del proceso ─────────────────────────────────────────────
    ELEMENT_TAG_MAP = {
        BPMNElementType.START_EVENT:       "startEvent",
        BPMNElementType.END_EVENT:         "endEvent",
        BPMNElementType.TASK:              "task",
        BPMNElementType.USER_TASK:         "userTask",
        BPMNElementType.SERVICE_TASK:      "serviceTask",
        BPMNElementType.EXCLUSIVE_GATEWAY: "exclusiveGateway",
        BPMNElementType.PARALLEL_GATEWAY:  "parallelGateway",
    }

    for elem in structure.elements:
        tag = ELEMENT_TAG_MAP.get(elem.type)
        if tag is None:
            continue

        attrib = {"id": elem.id, "name": elem.name}

        # Agregar marcador de evento de inicio/fin
        if elem.type == BPMNElementType.START_EVENT:
            elem_el = etree.SubElement(
                process_el,
                _make_tag(BPMN_NS, tag),
                attrib=attrib,
            )
            for _fl in structure.sequence_flows:
                if _fl.source_ref == elem.id:
                    _out = etree.SubElement(elem_el, _make_tag(BPMN_NS, "outgoing"))
                    _out.text = _fl.id
                    break

        elif elem.type == BPMNElementType.END_EVENT:
            elem_el = etree.SubElement(
                process_el,
                _make_tag(BPMN_NS, tag),
                attrib=attrib,
            )
            for _fl in structure.sequence_flows:
                if _fl.target_ref == elem.id:
                    _inc = etree.SubElement(elem_el, _make_tag(BPMN_NS, "incoming"))
                    _inc.text = _fl.id
                    break

        else:
            elem_el = etree.SubElement(
                process_el,
                _make_tag(BPMN_NS, tag),
                attrib=attrib,
            )
            for _fl in structure.sequence_flows:
                if _fl.target_ref == elem.id:
                    _inc = etree.SubElement(elem_el, _make_tag(BPMN_NS, "incoming"))
                    _inc.text = _fl.id
            for _fl in structure.sequence_flows:
                if _fl.source_ref == elem.id:
                    _out = etree.SubElement(elem_el, _make_tag(BPMN_NS, "outgoing"))
                    _out.text = _fl.id

        # Documentación
        if elem.documentation:
            doc_el = etree.SubElement(
                elem_el,
                _make_tag(BPMN_NS, "documentation"),
            )
            doc_el.text = elem.documentation

    # ── Sequence Flows ────────────────────────────────────────────────────
    for flow in structure.sequence_flows:
        flow_attrib = {
            "id":        flow.id,
            "sourceRef": flow.source_ref,
            "targetRef": flow.target_ref,
        }
        if flow.name:
            flow_attrib["name"] = flow.name

        etree.SubElement(
            process_el,
            _make_tag(BPMN_NS, "sequenceFlow"),
            attrib=flow_attrib,
        )

    # ── BPMNDI (posicionamiento automático del diagrama) ──────────────────
    diagram = etree.SubElement(
        root,
        _make_tag(BPMNDI_NS, "BPMNDiagram"),
        attrib={"id": "BPMNDiagram_1"},
    )
    bpmn_plane = etree.SubElement(
        diagram,
        _make_tag(BPMNDI_NS, "BPMNPlane"),
        attrib={
            "id":         "BPMNPlane_1",
            "bpmnElement": "Collaboration_1",
        },
    )

    # Posicionamiento automático en cascada horizontal
    x_offset = 150
    y_base   = 200
    x_step   = 200

    for i, elem in enumerate(structure.elements):
        if elem.type in (
            BPMNElementType.SEQUENCE_FLOW,
        ):
            continue

        shape_attrib = {
            "id":          f"Shape_{elem.id}",
            "bpmnElement": elem.id,
        }
        if elem.type in (
            BPMNElementType.EXCLUSIVE_GATEWAY,
            BPMNElementType.PARALLEL_GATEWAY,
        ):
            shape_attrib["isMarkerVisible"] = "true"

        shape = etree.SubElement(
            bpmn_plane,
            _make_tag(BPMNDI_NS, "BPMNShape"),
            attrib=shape_attrib,
        )

        # Dimensiones por tipo de elemento
        if elem.type in (BPMNElementType.START_EVENT, BPMNElementType.END_EVENT):
            w, h = 36, 36
        elif elem.type in (
            BPMNElementType.EXCLUSIVE_GATEWAY,
            BPMNElementType.PARALLEL_GATEWAY,
        ):
            w, h = 50, 50
        else:
            w, h = 120, 80

        bounds = etree.SubElement(
            shape,
            _make_tag(DC_NS, "Bounds"),
            attrib={
                "x":      str(x_offset + i * x_step),
                "y":      str(y_base),
                "width":  str(w),
                "height": str(h),
            },
        )

    # Edges con waypoints calculados
    elem_positions = {}
    visible = [e for e in structure.elements if e.type.name != "SEQUENCE_FLOW"]
    for i, e in enumerate(visible):
        w, h = (36,36) if e.type.name in ("START_EVENT","END_EVENT") else (50,50) if e.type.name in ("EXCLUSIVE_GATEWAY","PARALLEL_GATEWAY") else (120,80)
        elem_positions[e.id] = (150 + i*200 + w/2, 200 + h/2)

    for flow in structure.sequence_flows:
        edge = etree.SubElement(bpmn_plane, _make_tag(BPMNDI_NS, "BPMNEdge"),
            attrib={"id": f"Edge_{flow.id}", "bpmnElement": flow.id})
        src = elem_positions.get(flow.source_ref)
        tgt = elem_positions.get(flow.target_ref)
        if src and tgt:
            wp1 = etree.SubElement(edge, _make_tag(DI_NS, "waypoint"))
            wp1.set("x", str(int(src[0]))); wp1.set("y", str(int(src[1])))
            wp2 = etree.SubElement(edge, _make_tag(DI_NS, "waypoint"))
            wp2.set("x", str(int(tgt[0]))); wp2.set("y", str(int(tgt[1])))

    # ── Serialización XML ─────────────────────────────────────────────────
    xml_bytes = etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )
    return xml_bytes.decode("utf-8")


# ─────────────────────────────────────────────
# PERSISTENCIA DEL ARCHIVO .bpmn
# ─────────────────────────────────────────────

def _save_bpmn_file(xml_content: str, process_name: str) -> str:
    """
    Guarda el XML en storage/outputs/bpmn/<process_name>_<timestamp>.bpmn
    Retorna la ruta absoluta del archivo guardado.
    """
    output_dir = Path(settings.bpmn_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nombre de archivo seguro
    safe_name = (
        process_name.lower()
        .replace(" ", "_")
        .replace("/", "-")
        [:50]
    )
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename  = f"{safe_name}_{timestamp}.bpmn"
    file_path = output_dir / filename

    file_path.write_text(xml_content, encoding="utf-8")
    logger.info(f"BPMN guardado: {file_path}")
    return str(file_path)


# ─────────────────────────────────────────────
# VALIDACIÓN DEL XML GENERADO
# ─────────────────────────────────────────────

def _validate_bpmn_xml(xml_content: str) -> list[str]:
    """
    Valida el XML BPMN generado.
    Retorna lista de errores (vacía = válido).
    """
    errors: list[str] = []

    try:
        root = etree.fromstring(xml_content.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        return [f"XML inválido: {str(e)}"]

    ns = {"bpmn": BPMN_NS}

    # Verificar que existe exactamente 1 startEvent
    start_events = root.findall(".//bpmn:startEvent", ns)
    if len(start_events) != 1:
        errors.append(
            f"Se esperaba 1 startEvent, se encontraron {len(start_events)}"
        )

    # Verificar que existe exactamente 1 endEvent
    end_events = root.findall(".//bpmn:endEvent", ns)
    if len(end_events) != 1:
        errors.append(
            f"Se esperaba 1 endEvent, se encontraron {len(end_events)}"
        )

    # Verificar que existen sequenceFlows
    flows = root.findall(".//bpmn:sequenceFlow", ns)
    if len(flows) == 0:
        errors.append("El BPMN no tiene sequenceFlows — el proceso no está conectado")

    # Verificar que existen tareas
    tasks = (
        root.findall(".//bpmn:task", ns) +
        root.findall(".//bpmn:userTask", ns) +
        root.findall(".//bpmn:serviceTask", ns)
    )
    if len(tasks) == 0:
        errors.append("El BPMN no tiene tareas (task/userTask/serviceTask)")

    return errors


# ─────────────────────────────────────────────
# NODO LANGGRAPH
# ─────────────────────────────────────────────

def node_generate_bpmn(state: AgentState) -> dict:
    """
    Nodo LangGraph: genera el XML BPMN 2.0 del proceso TO-BE aprobado.

    Entrada del estado: tobe_process, hitl_approved
    Salida al estado:   bpmn_output, bpmn_ok, errors
    """
    logger.info("── Nodo: generate_bpmn ──")

    tobe = state.tobe_process
    if tobe is None:
        return {
            "bpmn_ok": False,
            "errors":  state.errors + ["generate_bpmn: tobe_process es None."],
            "current_node": "generate_bpmn",
        }

    # Verificar aprobación HITL antes de generar
    if settings.hitl_enabled and not state.hitl_approved and state.hitl_retries < 2:
        return {
            "bpmn_ok": False,
            "errors":  state.errors + [
                "generate_bpmn: el TO-BE no ha sido aprobado por el revisor humano."
            ],
            "current_node": "generate_bpmn",
        }

    try:
        # ── 1. Construir estructura BPMN desde TO-BE (determinístico) ─────
        structure = build_bpmn_structure_from_tobe(tobe)

        # ── 2. Generar XML BPMN 2.0 ───────────────────────────────────────
        xml_content = _build_xml_from_structure(structure)

        # ── 3. Validar XML generado ───────────────────────────────────────
        validation_errors = _validate_bpmn_xml(xml_content)
        if validation_errors:
            logger.error(f"Errores de validación BPMN: {validation_errors}")
            return {
                "bpmn_ok": False,
                "errors":  state.errors + validation_errors,
                "current_node": "generate_bpmn",
            }

        # ── 4. Persistir archivo .bpmn ────────────────────────────────────
        file_path = _save_bpmn_file(xml_content, tobe.name)

        bpmn_output = BPMNOutput(
            process_id=tobe.id,
            process_name=tobe.name,
            xml_content=xml_content,
            file_path=file_path,
            element_count=len(structure.elements),
        )

        logger.info(
            f"BPMN generado exitosamente: {len(structure.elements)} elementos, "
            f"{len(structure.sequence_flows)} flujos — {file_path}"
        )

        return {
            "bpmn_output":  bpmn_output,
            "bpmn_ok":      True,
            "current_node": "generate_bpmn",
        }

    except Exception as e:
        logger.error(f"Error en generate_bpmn: {e}")
        return {
            "bpmn_ok": False,
            "errors":  state.errors + [f"generate_bpmn: {str(e)}"],
            "current_node": "generate_bpmn",
        }