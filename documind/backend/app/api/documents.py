"""
Document endpoints — FR2 (upload), FR3 (ingestion), FR10 (delete).
Upload triggers the full ingestion pipeline synchronously (parse -> chunk ->
embed -> store). Fine for MVP file sizes; a background queue is a Step-past-MVP
upgrade noted in docs/REQUIREMENTS.md if upload latency becomes an issue.
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.db.models import User, Document
from app import vectorstore
from app.schemas import DocumentResponse
from app.ingestion.parser import parse_document
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_chunks

router = APIRouter()

SUPPORTED_TYPES = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html"}


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    suffix = Path(file.filename).suffix.lower()
    doc_type = SUPPORTED_TYPES.get(suffix)
    if doc_type is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_TYPES)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = parse_document(tmp_path, doc_type)
        chunks = chunk_text(text, doc_type)
        if not chunks:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No extractable content in this file")

        embeddings = embed_chunks([c.content for c in chunks])

        document = Document(owner_id=user.id, filename=file.filename, doc_type=doc_type)
        db.add(document)
        db.flush()  # assigns document.id without committing yet

        # pgvector: adds Chunk rows to this session. chroma: writes to the local
        # Chroma collection (no Chunk rows in Postgres).
        vectorstore.add_chunks(
            document.id, user.id, file.filename, chunks, embeddings, db=db
        )
        try:
            db.commit()
        except Exception:
            db.rollback()
            vectorstore.delete_document(document.id, db=db)  # drop orphaned vectors
            raise
        db.refresh(document)
        return document
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Document).filter(Document.owner_id == user.id).order_by(Document.created_at.desc()).all()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_id == user.id)
        .first()
    )
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    vectorstore.delete_document(document.id, db=db)
    db.delete(document)
    db.commit()
