"""
Query endpoints — FR4 (ask question), FR5 (grounded answer + citations),
FR9 (query history). Runs the LangGraph agent graph (app/agents/graph.py).
"""
import logging
import time
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.deps import get_current_user
from app.db.session import get_db
from app.db.models import User, QueryLog
from app.schemas import AskRequest, AskResponse
from app.agents.graph import build_graph

router = APIRouter()
logger = logging.getLogger("documind.query")

# Compiled once per process — LangGraph graphs are stateless/reusable across invocations.
_graph = build_graph()

# Bounds on what earlier turns cost in tokens.
MAX_HISTORY_QUESTION_CHARS = 500
MAX_HISTORY_ANSWER_CHARS = 1200


def load_history(db: Session, user_id, conversation_id) -> list[dict]:
    """
    Last N question/answer pairs of this conversation, oldest first. Always
    filtered by user_id too, so a guessed conversation_id never exposes
    another user's turns. Stored server-side (not sent by the client), so a
    caller can't forge earlier 'assistant' turns.
    """
    limit = get_settings().history_turns
    if limit <= 0:
        return []
    rows = (
        db.query(QueryLog)
        .filter(
            QueryLog.user_id == user_id,
            QueryLog.conversation_id == conversation_id,
            QueryLog.answer.isnot(None),
        )
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "question": r.question[:MAX_HISTORY_QUESTION_CHARS],
            "answer": r.answer[:MAX_HISTORY_ANSWER_CHARS],
        }
        for r in reversed(rows)
    ]


@router.post("/ask", response_model=AskResponse)
def ask_question(
    body: AskRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    started = time.perf_counter()
    conversation_id = body.conversation_id or uuid.uuid4()
    history = load_history(db, user.id, conversation_id)
    logger.info(
        "ask start | user=%s conversation=%s history_turns=%d question_chars=%d",
        user.id, conversation_id, len(history), len(body.question),
    )
    result = _graph.invoke(
        {
            "question": body.question,
            "user_id": str(user.id),
            "history": history,
            "trace": [],
            "date_from": body.uploaded_after.isoformat() if body.uploaded_after else None,
            "date_to": body.uploaded_before.isoformat() if body.uploaded_before else None,
        }
    )
    logger.info(
        "ask done  | user=%s %.0f ms retrieval_attempts=%s citations=%d answer_chars=%d",
        user.id, (time.perf_counter() - started) * 1000,
        result.get("retrieval_attempts"), len(result["citations"]), len(result["answer"]),
    )

    log = QueryLog(
        user_id=user.id, conversation_id=conversation_id, question=body.question, answer=result["answer"]
    )
    db.add(log)
    db.commit()

    return AskResponse(
        question=body.question,
        conversation_id=conversation_id,
        answer=result["answer"],
        citations=result["citations"],
        trace=result["trace"],
    )


@router.get("/history", response_model=list[AskResponse])
def get_query_history(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    logs = (
        db.query(QueryLog)
        .filter(QueryLog.user_id == user.id)
        .order_by(QueryLog.created_at.desc())
        .limit(50)
        .all()
    )
    # History only persists question/answer (see db/models.py QueryLog) —
    # citations/trace aren't stored per-row yet, so they're empty on replay.
    return [
        AskResponse(
            question=log.question,
            conversation_id=log.conversation_id,
            answer=log.answer or "",
            citations=[],
            trace=[],
        )
        for log in logs
    ]
