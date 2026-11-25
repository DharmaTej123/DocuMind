"""pgvector backend: logic moved faithfully from api/documents.py and agents/retriever.py."""
from app.db.models import Chunk, Document
from app.db.session import SessionLocal


def add_chunks(document_id, owner_id, filename, chunks, embeddings, db=None) -> None:
    # Adds to the caller's session (caller commits). owner_id/filename are
    # derived from the Document row in this backend, so they are unused here.
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


def search(query_embedding, user_id, top_k) -> list[dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Document.owner_id == user_id)
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
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
