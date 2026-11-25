"""pgvector backend: logic moved faithfully from api/documents.py and agents/retriever.py."""
from app.db.models import Chunk, Document
from app.db.session import SessionLocal
from app.vectorstore.dates import from_epoch, to_epoch


def add_chunks(
    document_id, owner_id, filename, chunks, embeddings, db=None, doc_meta=None, uploaded_at=None
) -> None:
    # Adds to the caller's session (caller commits). owner_id, filename, doc_meta
    # and the upload time live on the Document row in this backend (created_at),
    # so they are unused here.
    own = db is None
    session = db or SessionLocal()
    try:
        for chunk, embedding in zip(chunks, embeddings):
            session.add(
                Chunk(
                    document_id=document_id,
                    content=chunk.content,
                    embedding=embedding,
                    section=chunk.section,
                )
            )
        if own:
            session.commit()
    finally:
        if own:
            session.close()


def delete_document(document_id, db=None) -> None:
    own = db is None
    session = db or SessionLocal()
    try:
        session.query(Chunk).filter(Chunk.document_id == document_id).delete()
        if own:
            session.commit()
    finally:
        if own:
            session.close()


def list_chunks(user_id) -> list[dict]:
    """All of one user's chunks (no embeddings), for keyword indexing."""
    db = SessionLocal()
    try:
        rows = (
            db.query(Chunk, Document.filename, Document.created_at)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Document.owner_id == user_id)
            .all()
        )
    finally:
        db.close()
    return [
        {
            "chunk_id": str(c.id),
            "content": c.content,
            "section": c.section,
            "doc": fn,
            "uploaded_ts": to_epoch(created) if created else None,
        }
        for c, fn, created in rows
    ]


def search(query_embedding, user_id, top_k, ts_from=None, ts_to=None) -> list[dict]:
    db = SessionLocal()
    try:
        query = (
            db.query(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Document.owner_id == user_id)
        )
        # created_at is a naive UTC datetime; bounds are [from, to) epoch seconds.
        if ts_from is not None:
            query = query.filter(Document.created_at >= from_epoch(ts_from))
        if ts_to is not None:
            query = query.filter(Document.created_at < from_epoch(ts_to))
        rows = (
            query.order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
            .all()
        )
    finally:
        db.close()

    return [
        {
            "chunk_id": str(chunk.id),
            "content": chunk.content,
            "section": chunk.section,
            "doc": filename,
        }
        for chunk, filename in rows
    ]
