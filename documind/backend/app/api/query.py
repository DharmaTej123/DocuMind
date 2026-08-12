"""
Query endpoints — FR4 (ask question), FR5 (grounded answer + citations),
FR9 (query history). Runs the LangGraph agent graph (app/agents/graph.py).
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.db.models import User, QueryLog
from app.schemas import AskRequest, AskResponse
from app.agents.graph import build_graph

router = APIRouter()

# Compiled once per process — LangGraph graphs are stateless/reusable across invocations.
_graph = build_graph()


@router.post("/ask", response_model=AskResponse)
def ask_question(
    body: AskRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = _graph.invoke(
        {
            "question": body.question,
            "user_id": str(user.id),
            "trace": [],
        }
    )

    log = QueryLog(user_id=user.id, question=body.question, answer=result["answer"])
    db.add(log)
    db.commit()

    return AskResponse(
        question=body.question,
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
        AskResponse(question=log.question, answer=log.answer or "", citations=[], trace=[])
        for log in logs
    ]
