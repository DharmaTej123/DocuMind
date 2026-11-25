"""
Embedding generation for chunks before pgvector storage.

Provider-switchable via settings.llm_provider, same pattern as
core/llm_client.py. Both providers are configured to output
settings.embedding_dim (1536) so switching providers never requires a
pgvector schema migration:
  - OpenAI's text-embedding-3-small is natively 1536-dim.
  - Gemini's gemini-embedding-001 is natively 3072-dim but supports an
    output_dimensionality param to truncate — set to 1536 to match.
"""
import os
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)
from app.core.config import get_settings
import voyageai
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Batch-embeds chunk texts. Returns one vector per input chunk, in order."""
    if not chunks:
        return []

    settings = get_settings()
    # Groq and Anthropic have no embeddings API, so both use VoyageAI.
    if settings.llm_provider in ("groq", "anthropic"):
        return _embed_chunks_voyageai(chunks, settings)
    # if settings.llm_provider == "gemini":
    #     return _embed_chunks_gemini(chunks, settings)
    if settings.llm_provider == "openai":
        return _embed_chunks_openai(chunks, settings)
    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")

def _embed_chunks_voyageai(chunks: list[str], settings) -> list[list[float]]:
    client = voyageai.Client(api_key=settings.voyageai_api_key)

    response = client.embed(
        model=settings.voyageai_embedding_model,
        texts=chunks,
        output_dimension=settings.embedding_dim
    )
    return response.embeddings
def _embed_chunks_openai(chunks: list[str], settings) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url="https://openai.vocareum.com/v1",  # same gateway as llm_client.py
    )
    # text-embedding-3-small is natively 1536-dim but supports `dimensions`;
    # request settings.embedding_dim so vectors match the Vector(512) column.
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=chunks,
        dimensions=settings.embedding_dim,
    )
    return [item.embedding for item in response.data]


def _embed_chunks_gemini(chunks: list[str], settings) -> list[list[float]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    # Gemini's embed_content takes one input per call as of gemini-embedding-001
    # (see docs/GEMINI_MIGRATION.md) — batch by looping rather than a single
    # multi-input request.
    vectors = []
    for chunk in chunks:
        result = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=chunk,
            config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dim),
        )
        vectors.append(result.embeddings[0].values)
    return vectors


def embed_query(query: str) -> list[float]:
    """Embeds a single query string — used by the retriever node."""
    return embed_chunks([query])[0]
