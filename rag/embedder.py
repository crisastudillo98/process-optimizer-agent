#from __future__ import annotations
#from langchain_openai import OpenAIEmbeddings
#from config.settings import settings
#from observability.logger import get_logger

#logger = get_logger(__name__)


#def get_embedder() -> OpenAIEmbeddings:
#    """
#    Retorna el modelo de embeddings configurado.
#    Singleton liviano — LangChain maneja el pool de conexiones internamente.
#    """
#    logger.info(f"Embedder inicializado: {settings.embedding_model}")
#    return OpenAIEmbeddings(
#        model=settings.embedding_model,
#        api_key=settings.openai_api_key,
#    )

# rag/embedder.py
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

def get_embedder():
    return DefaultEmbeddingFunction()

__all__ = ["get_embedder"]