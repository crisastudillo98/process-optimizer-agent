from __future__ import annotations
from config.settings import settings
from observability.logger import get_logger

logger = get_logger(__name__)


def get_llm(temperature: float | None = None):
    """
    Factory centralizado de LLMs.
    Lee el provider desde settings y retorna el cliente configurado.
    Soporta: OpenAI, Groq, Perplexity.
    """
    temp = temperature if temperature is not None else settings.openai_temperature

    provider = settings.llm_provider
    logger.info(f"LLM factory: provider={provider}, temperature={temp}")

    # ── Groq ──────────────────────────────────────────────────────────────
    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY no está configurada en el .env. "
                "Obtén tu key gratis en https://console.groq.com"
            )
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=settings.groq_model,
            temperature=temp,
            api_key=settings.groq_api_key,
        )
        logger.info(f"LLM inicializado: Groq / {settings.groq_model}")
        return llm

    # ── OpenAI (default) ──────────────────────────────────────────────────
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY no está configurada en el .env. "
            "O cambia LLM_PROVIDER=groq en tu .env para usar Groq gratis."
        )
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=temp,
        api_key=settings.openai_api_key,
    )
    logger.info(f"LLM inicializado: OpenAI / {settings.openai_model}")
    return llm


def get_embedder():
    """
    Factory centralizado de embeddings.
    Usa ChromaDB DefaultEmbeddingFunction — sin API key,
    sin descarga manual, sin problemas de permisos en Docker.
    Internamente usa all-MiniLM-L6-v2 via sentence-transformers.
    """
    logger.info("Embedder factory: ChromaDB DefaultEmbeddingFunction")
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    embedder = DefaultEmbeddingFunction()
    logger.info("Embedder inicializado: ChromaDB / all-MiniLM-L6-v2")
    return embedder