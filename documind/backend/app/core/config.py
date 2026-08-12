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
    llm_provider: str = "groq"

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
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Embedding vector dimension stored in pgvector (db/models.py EMBEDDING_DIM).
    # Gemini's gemini-embedding-001 natively outputs 3072-dim vectors but
    # supports truncating via output_dimensionality — set to 1536 so it's a
    # drop-in match for OpenAI's text-embedding-3-small dimension, meaning
    # no schema migration is needed when switching providers.
    embedding_dim: int = 512

    # Rerank
    cohere_api_key: str = ""

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "documind"
    langsmith_tracing: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
