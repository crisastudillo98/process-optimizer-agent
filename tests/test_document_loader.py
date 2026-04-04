import pytest
from pathlib import Path
from agent.document_loader import load_document, node_load_document
from models.schemas import AgentState


def test_load_plain_text():
    text = load_document("El proceso inicia cuando el cliente realiza un pedido.")
    assert "proceso" in text.lower()


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_document("/ruta/inexistente/archivo.pdf")


def test_load_unsupported_extension(tmp_path):
    file = tmp_path / "doc.csv"
    file.write_text("col1,col2")
    with pytest.raises(ValueError, match="no soportada"):
        load_document(file)


def test_load_txt_file(tmp_path):
    file = tmp_path / "proceso.txt"
    file.write_text("El proceso de facturación tiene 5 pasos principales.")
    result = load_document(file)
    assert "facturación" in result


def test_node_load_document_text_input():
    state = AgentState(raw_input="Proceso de compras con 4 actividades.")
    result = node_load_document(state)
    assert "raw_input" in result
    assert "compras" in result["raw_input"]


def test_node_load_document_empty_input():
    state = AgentState(raw_input="", input_file_path=None)
    result = node_load_document(state)
    assert len(result["errors"]) > 0


def test_node_load_document_short_content():
    state = AgentState(raw_input="corto")
    result = node_load_document(state)
    assert len(result["errors"]) > 0
    assert "caracteres" in result["errors"][0]