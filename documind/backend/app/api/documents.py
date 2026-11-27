"""
Document endpoints — FR2 (upload), FR3 (ingestion), FR10 (delete).
Upload triggers the full ingestion pipeline synchronously (parse -> chunk ->
embed -> store). Fine for MVP file sizes; a background queue is a Step-past-MVP
upgrade noted in docs/REQUIREMENTS.md if upload latency becomes an issue.
"""
import logging
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.db.models import User, Document
from app import vectorstore
from app.schemas import DocumentResponse
from app.ingestion.parser import parse_document
from app.ingestion.chunker import add_context, chunk_text
from app.ingestion.embedder import embed_chunks
from app.ingestion.metadata import extract_doc_metadata
from app.core.config import get_settings

router = APIRouter()
logger = logging.getLogger("documind.documents")

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

    started = time.perf_counter()
    logger.info("upload start | user=%s file=%s type=%s", user.id, file.filename, doc_type)
    try:
        text = parse_document(tmp_path, doc_type)
        # One LLM call per document (not per chunk); {} if disabled or on any failure.
        doc_meta = extract_doc_metadata(text, file.filename) if get_settings().extract_doc_metadata else {}
        chunks = add_context(chunk_text(text, doc_type), file.filename, doc_meta)
        tables = sum(1 for c in chunks if (c.section or "").startswith("Table"))
        ocr_chunks = sum(1 for c in chunks if (c.section or "").startswith("OCR"))
        logger.info(
            "upload parsed | file=%s text_chars=%d chunks=%d table_chunks=%d ocr_chunks=%d",
            file.filename, len(text), len(chunks), tables, ocr_chunks,
        )
        if not chunks:
            logger.warning(
                "upload rejected | file=%s has no extractable text; if it is a scan, check that OCR "
                "is installed and OCR_ENABLED is true (see the parser/ocr log lines above)",
                file.filename,
            )
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "No extractable text in this file. If it is a scanned PDF, OCR must be installed and enabled.",
            )

        embeddings = embed_chunks([c.content for c in chunks])
        logger.info("upload embedded | file=%s vectors=%d dim=%d", file.filename, len(embeddings), len(embeddings[0]))

        document = Document(
            owner_id=user.id, filename=file.filename, doc_type=doc_type, doc_metadata=doc_meta or None
        )
        db.add(document)
        db.flush()  # assigns document.id without committing yet

        # pgvector: adds Chunk rows to this session. chroma: writes to the local
        # Chroma collection (no Chunk rows in Postgres).
        vectorstore.add_chunks(
            document.id, user.id, file.filename, chunks, embeddings, db=db,
            doc_meta=doc_meta, uploaded_at=document.created_at,
        )
        try:
            db.commit()
        except Exception:
            db.rollback()
            vectorstore.delete_document(document.id, db=db)  # drop orphaned vectors
            raise
        db.refresh(document)
        logger.info(
            "upload done  | file=%s document_id=%s store=%s %.0f ms",
            file.filename, document.id, vectorstore.active_store(), (time.perf_counter() - started) * 1000,
        )
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
