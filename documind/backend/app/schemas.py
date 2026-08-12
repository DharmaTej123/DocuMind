"""
Pydantic schemas for API request/response bodies. Kept separate from
db/models.py (SQLAlchemy ORM models) on purpose — API shape and storage
shape are allowed to diverge.
"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# --- Auth ---
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RegisterResponse(BaseModel):
    id: UUID


# --- Documents ---
class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    doc_type: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Query ---
class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class Citation(BaseModel):
    doc: str
    section: Optional[str] = None


class TraceStage(BaseModel):
    stage: str  # interpret | retrieve | rerank | synthesize
    summary: str
    detail: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    trace: List[TraceStage]
