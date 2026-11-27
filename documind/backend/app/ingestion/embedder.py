"""
Embedding generation for chunks before pgvector storage.

Provider-switchable via settings.llm_provider, same pattern as
core/llm_client.py. Every provider is asked for settings.embedding_dim
dimensions (512, matching the Vector(512) column), but vectors from different
providers are NOT comparable: re-upload documents after switching provider.
  - VoyageAI (groq, anthropic): output_dimension=embedding_dim.
  - OpenAI text-embedding-3-small: `dimensions=embedding_dim`.
  - Gemini gemini-embedding-001: output_dimensionality=embedding_dim.
"""
import os
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)
from app.core.config import LLMProvider, get_settings
import voyageai
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Batch-embeds chunk texts. Returns one vector per input chunk, in order."""
    if not chunks:
        return []

    settings = get_settings()
    return _EMBEDDERS[settings.llm_provider](chunks, settings)

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


# Groq and Anthropic have no embeddings API, so both use VoyageAI. Keyed by the
# LLMProvider enum; the check below fails at import if a provider has no embedder.
_EMBEDDERS = {
    LLMProvider.GROQ: _embed_chunks_voyageai,
    LLMProvider.ANTHROPIC: _embed_chunks_voyageai,
    LLMProvider.OPENAI: _embed_chunks_openai,
    LLMProvider.GEMINI: _embed_chunks_gemini,
}

_missing = set(LLMProvider) - set(_EMBEDDERS)
if _missing:
    raise RuntimeError(f"embedder has no embedding function for provider(s): {sorted(p.value for p in _missing)}")


def embed_query(query: str) -> list[float]:
    """Embeds a single query string — used by the retriever node."""
    return embed_chunks([query])[0]
