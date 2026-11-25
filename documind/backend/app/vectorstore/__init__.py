"""
Vector store abstraction. Selected by settings.vector_store:

  - "pgvector" (default): chunks + embeddings live in the Postgres `chunks` table.
  - "chroma" (TEMPORARY local alternative): chunk text + embeddings live in a
    persistent local Chroma collection. Postgres still holds Document rows (and
    users); no Chunk rows are written in this mode.

Existing data is NOT migrated between stores. Re-upload documents after switching.

Public API (all take precomputed embeddings; Chroma never embeds anything itself):
    add_chunks(document_id, owner_id, filename, chunks, embeddings, db=None)
    delete_document(document_id, db=None)
    search(query_embedding, user_id, top_k) -> list[{chunk_id, content, section, doc}]

`db` is an optional SQLAlchemy session. The pgvector backend uses it so chunk
inserts/deletes join the caller's transaction (the caller commits). The Chroma
backend ignores it.
"""
from app.core.config import get_settings

VALID_STORES = ("pgvector", "chroma")


def _backend():
    store = get_settings().vector_store
    if store == "pgvector":
        from app.vectorstore import pgvector_store
        return pgvector_store
    if store == "chroma":
        from app.vectorstore import chroma_store
        return chroma_store
    raise ValueError(f"Unknown VECTOR_STORE '{store}'. Allowed values: {list(VALID_STORES)}")


def active_store() -> str:
    """Name of the active store; raises ValueError for unknown values."""
    _backend()
    return get_settings().vector_store


def add_chunks(document_id, owner_id, filename, chunks, embeddings, db=None) -> None:
    _backend().add_chunks(document_id, owner_id, filename, chunks, embeddings, db)


def delete_document(document_id, db=None) -> None:
    _backend().delete_document(document_id, db)


def search(query_embedding, user_id, top_k) -> list[dict]:
    return _backend().search(query_embedding, user_id, top_k)
