"""
Synthesis node — generates the final grounded answer from the reranked
chunks, with citations back to source doc + section (FR5).

Prompt layout (see agents/prompting.py): a system prompt with the rules and
data boundaries, earlier conversation turns as user/assistant messages, then
one user message holding the <context> chunks and the <question>. The model
replies NOT_FOUND when the chunks don't contain the answer; that is turned
into a plain "couldn't find it" message with no citations.
"""
from app.agents.prompting import (
    NOT_FOUND,
    NOT_FOUND_MESSAGE,
    build_answer_messages,
    message_text,
    warn_if_suspicious,
)
from app.core.config import get_settings
from app.core.llm_client import get_llm


def _no_answer(state: dict, answer: str, summary: str, detail: str) -> dict:
    return {
        "answer": answer,
        "citations": [],
        "trace": state.get("trace", []) + [{"stage": "synthesize", "summary": summary, "detail": detail}],
    }


def synthesize_node(state: dict) -> dict:
    question = state["question"]
    chunks = state["reranked_chunks"]
    history = state.get("history") or []
    settings = get_settings()

    if not chunks:
        # Nothing to ground an answer in (no documents, or none match the date
        # filter): answer plainly without an LLM call that could only guess.
        date_from, date_to = state.get("date_from"), state.get("date_to")
        why = (
            f" uploaded between {date_from or 'the beginning'} and {date_to or 'today'}"
            if date_from or date_to
            else ""
        )
        return _no_answer(
            state,
            f"I couldn't find any documents{why} to answer this question from.",
            "No chunks retrieved, so no answer was generated.",
            "retrieved_chunks was empty" + (", date filter active" if why else ""),
        )

    # Optional relevance gate (off by default): if even the best chunk scores below
    # the floor, say so without calling the LLM. Tune from the RERANK_DEBUG output.
    floor = settings.answer_min_rerank_score
    top = max(c.get("relevance_score", 0.0) for c in chunks)
    if floor > 0 and top < floor:
        return _no_answer(
            state,
            NOT_FOUND_MESSAGE,
            "Retrieved chunks were too weak to answer from.",
            f"top rerank score {top:.3f} < answer_min_rerank_score {floor}",
        )

    warn_if_suspicious(question, chunks)

    result = get_llm().invoke(build_answer_messages(question, chunks, history))
    answer = message_text(result)

    if answer.upper().startswith(NOT_FOUND):
        return _no_answer(
            state,
            NOT_FOUND_MESSAGE,
            "The model found no answer in the retrieved chunks.",
            f"model replied {NOT_FOUND}; {len(chunks)} chunks seen, citations withheld",
        )

    citations = [{"doc": c["doc"], "section": c["section"]} for c in chunks]
    trace_entry = {
        "stage": "synthesize",
        "summary": "Answer generated and grounded in the reranked chunks.",
        "detail": (
            f"model={getattr(settings, f'{settings.llm_provider}_model', settings.llm_provider)}, "
            f"citations_attached={len(citations)}, history_turns={len(history)}"
        ),
    }
    return {
        "answer": answer,
        "citations": citations,
        "trace": state.get("trace", []) + [trace_entry],
    }
