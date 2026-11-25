"""
Pydantic schemas for API request/response bodies. Kept separate from
db/models.py (SQLAlchemy ORM models) on purpose — API shape and storage
shape are allowed to diverge.
"""
from datetime import date, datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, model_validator


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
    doc_metadata: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Query ---
class AskRequest(BaseModel):
    # Length cap: very long inputs are a common vehicle for jailbreak attempts.
    question: str = Field(min_length=1, max_length=2000)
    # Send the conversation_id from a previous response to ask a follow-up.
    # Omit it to start a new conversation.
    conversation_id: Optional[UUID] = None
    # Only search documents uploaded in this range. Both optional, both inclusive
    # calendar days in UTC (e.g. "2026-01-01").
    uploaded_after: Optional[date] = None
    uploaded_before: Optional[date] = None

    @model_validator(mode="after")
    def _range_is_ordered(self):
        if self.uploaded_after and self.uploaded_before and self.uploaded_after > self.uploaded_before:
            raise ValueError("uploaded_after must be on or before uploaded_before")
        return self


class Citation(BaseModel):
    doc: str
    section: Optional[str] = None


class TraceStage(BaseModel):
    stage: str  # interpret | retrieve | rerank | synthesize
    summary: str
    detail: str


class AskResponse(BaseModel):
    question: str
    conversation_id: Optional[UUID] = None
    answer: str
    citations: List[Citation]
    trace: List[TraceStage]
