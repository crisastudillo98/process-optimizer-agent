from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.documents import Document

from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)

# Colecciones separadas por dominio de conocimiento
COLLECTION_PROCESS_CASES = "process_cases"
COLLECTION_LEAN_KB        = "lean_kb"

# Embedder compartido — instancia única
_embedder = DefaultEmbeddingFunction()


def _get_client() -> chromadb.ClientAPI:
    """Cliente ChromaDB persistente."""
    persist_path = str(Path(settings.vector_db_path))
    Path(persist_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_path)


def _get_collection(collection: str) -> chromadb.Collection:
    """Obtiene o crea una colección ChromaDB."""
    client = _get_client()
    return client.get_or_create_collection(
        name=collection,
        embedding_function=_embedder,
    )


# ─────────────────────────────────────────────
# OPERACIONES DE ESCRITURA
# ─────────────────────────────────────────────

def store_process_case(
    process_name: str,
    asis_summary: str,
    tobe_summary: str,
    industry: str = "general",
    improvements: list[str] | None = None,
) -> None:
    """
    Persiste un caso AS-IS → TO-BE en la Vector DB.
    """
    collection = _get_collection(COLLECTION_PROCESS_CASES)

    content = (
        f"PROCESO: {process_name}\n"
        f"INDUSTRIA: {industry}\n\n"
        f"AS-IS:\n{asis_summary}\n\n"
        f"TO-BE:\n{tobe_summary}\n\n"
        f"MEJORAS APLICADAS:\n" +
        "\n".join(f"- {m}" for m in (improvements or []))
    )

    import uuid
    collection.add(
        documents=[content],
        metadatas=[{
            "process_name": process_name,
            "industry":     industry,
            "type":         "process_case",
        }],
        ids=[str(uuid.uuid4())],
    )
    logger.info(f"Caso persistido en Vector DB: '{process_name}'")


def store_lean_knowledge(documents: list[Document]) -> None:
    """
    Indexa documentos Lean/Six Sigma/Kaizen.
    Se ejecuta una sola vez al inicializar (seed).
    """
    collection = _get_collection(COLLECTION_LEAN_KB)

    import uuid
    collection.add(
        documents=[doc.page_content for doc in documents],
        metadatas=[doc.metadata or {"type": "lean_kb"} for doc in documents],
        ids=[str(uuid.uuid4()) for _ in documents],
    )
    logger.info(f"{len(documents)} documentos Lean indexados en Vector DB")


# ─────────────────────────────────────────────
# OPERACIONES DE LECTURA
# ─────────────────────────────────────────────

def retrieve_similar_cases(query: str, k: int | None = None) -> list[Document]:
    """
    Recupera los k casos más similares a la consulta.
    """
    k = k or settings.rag_top_k
    collection = _get_collection(COLLECTION_PROCESS_CASES)

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count() or 1),
    )

    docs = [
        Document(
            page_content=content,
            metadata=meta or {},
        )
        for content, meta in zip(
            results["documents"][0],
            results["metadatas"][0],
        )
    ]
    logger.info(f"RAG casos: {len(docs)} resultados para query '{query[:60]}...'")
    return docs


def retrieve_lean_patterns(query: str, k: int | None = None) -> list[Document]:
    """
    Recupera patrones Lean/Six Sigma relevantes.
    """
    k = k or settings.rag_top_k
    collection = _get_collection(COLLECTION_LEAN_KB)

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count() or 1),
    )

    docs = [
        Document(
            page_content=content,
            metadata=meta or {},
        )
        for content, meta in zip(
            results["documents"][0],
            results["metadatas"][0],
        )
    ]
    logger.info(f"RAG Lean KB: {len(docs)} resultados")
    return docs


def get_collection_stats() -> dict:
    """Estadísticas de ambas colecciones para monitoreo."""
    cases_col = _get_collection(COLLECTION_PROCESS_CASES)
    lean_col  = _get_collection(COLLECTION_LEAN_KB)

    return {
        "process_cases_count": cases_col.count(),
        "lean_kb_count":       lean_col.count(),
        "vector_db_path":      settings.vector_db_path,
    }