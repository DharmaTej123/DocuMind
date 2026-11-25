"""
Chroma backend (TEMPORARY local alternative to pgvector).

Embeddings are always passed in explicitly (VoyageAI, 512-dim); Chroma's
default embedding function is never used. Each chunk carries owner_id metadata
and every search filters on it, preserving per-user isolation. The collection
uses cosine space, so returned distances match pgvector's cosine_distance
ordering (smaller = nearer); results are returned nearest-first.
"""
import uuid
from functools import lru_cache

from app.core.config import get_settings

COLLECTION_NAME = "documind_chunks"


@lru_cache
def _collection():
    import chromadb  # lazy: only needed when VECTOR_STORE=chroma

    client = chromadb.PersistentClient(path=get_settings().chroma_path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # we always supply embeddings
    )


def _where(user_id, ts_from=None, ts_to=None) -> dict:
    """Owner filter plus optional upload-time range on 'uploaded_ts' (epoch seconds)."""
    conditions = [{"owner_id": str(user_id)}]
    if ts_from is not None:
        conditions.append({"uploaded_ts": {"$gte": ts_from}})
    if ts_to is not None:
        conditions.append({"uploaded_ts": {"$lt": ts_to}})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def add_chunks(
    document_id, owner_id, filename, chunks, embeddings, db=None, doc_meta=None, uploaded_at=None
) -> None:
    if not chunks:
        return
    from datetime import datetime

    from app.ingestion.metadata import chroma_fields
    from app.vectorstore.dates import to_epoch

    extra = chroma_fields(doc_meta)
    # Integer epoch seconds: Chroma range filters only work on numbers.
    extra["uploaded_ts"] = to_epoch(uploaded_at or datetime.utcnow())
    _collection().add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        embeddings=[list(map(float, e)) for e in embeddings],
        documents=[c.content for c in chunks],
        metadatas=[
            {
                "document_id": str(document_id),
                "owner_id": str(owner_id),
                "filename": filename,
                "section": c.section or "",  # Chroma metadata cannot be None
                **extra,  # subject, kind, topics (strings only; Chroma rejects lists)
            }
            for c in chunks
        ],
    )


def delete_document(document_id, db=None) -> None:
    _collection().delete(where={"document_id": str(document_id)})


def list_chunks(user_id) -> list[dict]:
    """All of one user's chunks (no embeddings), for keyword indexing."""
    res = _collection().get(where={"owner_id": str(user_id)}, include=["documents", "metadatas"])
    return [
        {
            "chunk_id": cid,
            "content": content,
            "section": meta.get("section") or None,
            "doc": meta.get("filename"),
            "uploaded_ts": meta.get("uploaded_ts"),  # None for chunks stored before dates existed
        }
        for cid, content, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]


def search(query_embedding, user_id, top_k, ts_from=None, ts_to=None) -> list[dict]:
    col = _collection()
    total = col.count()
    if total == 0:
        return []
    res = col.query(
        query_embeddings=[list(map(float, query_embedding))],
        n_results=min(top_k, total),
        where=_where(user_id, ts_from, ts_to),
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns results already sorted by ascending distance (nearest first).
    return [
        {
            "chunk_id": cid,
            "content": content,
            "section": meta.get("section") or None,
            "doc": meta.get("filename"),
        }
        for cid, content, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ]
