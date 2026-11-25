"""
Centralized app configuration, loaded from environment variables.
Never hardcode secrets here — this file defines shape, not values.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "DocuMind"
    environment: str = "development"

    # Database
    database_url: str = "postgresql+psycopg://postgres:postgres@db:5432/documind"

    # Auth
    jwt_secret_key: str = "change-me-in-env"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24h

    # LLM (default provider: Gemini — see docs/REQUIREMENTS.md Decisions Log)
    llm_provider: str = "anthropic"

    # Anthropic (Claude). Anthropic has no embeddings API, so embeddings for
    # this provider come from VoyageAI (see ingestion/embedder.py).
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    #LLM Groq
    groq_api_key: str = ""
    groq_model: str = "groq/compound"
    #Embedding model for Groq (if used)
    voyageai_embedding_model: str = "voyage-4-lite"
    voyageai_api_key: str = ""
    # Kept for anyone who wants to switch llm_provider back to "openai"
    openai_api_key: str = "voc-210425417112667747992946abc9bc0d7b650.29552297"
    openai_model: str = "gpt-4o-mini"

    # Embedding vector dimension stored in pgvector (db/models.py EMBEDDING_DIM).
    # Gemini's gemini-embedding-001 natively outputs 3072-dim vectors but
    # supports truncating via output_dimensionality — set to 1536 so it's a
    # drop-in match for OpenAI's text-embedding-3-small dimension, meaning
    # no schema migration is needed when switching providers.
    embedding_dim: int = 512

    # Vector store: "pgvector" (default) or "chroma" (TEMPORARY local alternative).
    # Existing data is NOT migrated between stores; re-upload documents after
    # switching. Chroma stores vectors on disk at chroma_path.
    vector_store: str = "pgvector"
    chroma_path: str = "./chroma_data"

    # Rerank
    cohere_api_key: str = ""

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "documind"
    langsmith_tracing: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
