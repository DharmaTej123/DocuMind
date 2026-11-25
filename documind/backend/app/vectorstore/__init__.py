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
from app.vectorstore import dates, hybrid

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


def add_chunks(
    document_id, owner_id, filename, chunks, embeddings, db=None, doc_meta=None, uploaded_at=None
) -> None:
    _backend().add_chunks(document_id, owner_id, filename, chunks, embeddings, db, doc_meta, uploaded_at)
    hybrid.invalidate(owner_id)


def delete_document(document_id, db=None) -> None:
    _backend().delete_document(document_id, db)
    hybrid.invalidate()  # owner unknown here, so drop all cached keyword indexes


def search(
    query_embedding, user_id, top_k, query_text=None, date_from=None, date_to=None
) -> list[dict]:
    """
    Vector search; when settings.hybrid_search is on and query_text is given,
    also runs BM25 keyword search over the user's chunks and fuses both
    rankings (RRF). Results gain a 'retrieved_via' key in hybrid mode.

    date_from / date_to: optional ISO 'YYYY-MM-DD' strings, inclusive days (UTC),
    restricting results to documents uploaded in that range. Applied to both the
    vector and keyword sides. Raises ValueError for a malformed date.
    """
    ts_from, ts_to = dates.bounds(date_from, date_to)
    backend = _backend()
    vector_hits = backend.search(query_embedding, user_id, top_k, ts_from, ts_to)
    if not query_text or not get_settings().hybrid_search:
        return vector_hits
    keyword_hits = hybrid.keyword_search(
        backend.list_chunks, user_id, query_text, top_k, ts_from, ts_to
    )
    return hybrid.fuse(vector_hits, keyword_hits, top_k)
