from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

class Settings(BaseSettings):
    # LLM Provider
    llm_provider: Literal["openai", "groq", "perplexity"] = Field(default="groq", env="LLM_PROVIDER")  
    
    # OpenAI opcional
    #openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.2, env="OPENAI_TEMPERATURE")

    # Groq
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")

    # RAG / Embeddings
    #embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_model: str = Field(default="local", env="EMBEDDING_MODEL")
    vector_db_path: str = Field(default="storage/vector_db", env="VECTOR_DB_PATH")
    rag_top_k: int = Field(default=5, env="RAG_TOP_K")

    # BPMN
    bpmn_output_path: str = Field(default="storage/outputs/bpmn", env="BPMN_OUTPUT_PATH")
    reports_output_path: str = Field(default="storage/outputs/reports", env="REPORTS_OUTPUT_PATH")

    # Observabilidad
    langsmith_api_key: str = Field(default="", env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="process-optimizer", env="LANGSMITH_PROJECT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Database
    database_url: str = Field(default="sqlite:///storage/process_optimizer.db", env="DATABASE_URL")

    # HITL
    hitl_enabled: bool = Field(default=True, env="HITL_ENABLED")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()