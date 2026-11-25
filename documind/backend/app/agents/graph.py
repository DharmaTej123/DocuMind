"""
LangGraph agent graph — FR7.

Flow: interpret_query -> retrieve -> rerank -> [route] -> synthesize -> END
                              ^_______________________|
The route after rerank sends the graph back to retrieve (with a wider
top_k) if the reranked results look weak, up to MAX_RETRIEVAL_ATTEMPTS —
otherwise it proceeds to synthesize. This is an actual conditional
retrieval loop, not just a linear pipeline: a low-confidence first pass on
an ambiguous question gets one more chance with a broader search before
answering.

Each node appends one entry to state["trace"] describing what it did — this
trace is returned to the frontend as-is and rendered by the AgentTrace
component, so the UI shows the real pipeline, not a simulated one.
"""
from typing import TypedDict, List, Optional
import logging
import time
from langgraph.graph import StateGraph, END

logger = logging.getLogger("documind.agent")

MAX_RETRIEVAL_ATTEMPTS = 2
# Retry threshold lives in settings.rerank_confidence_threshold (default 0.1).


class AgentState(TypedDict):
    question: str
    user_id: str
    history: List[dict]  # earlier turns of this conversation: [{question, answer}], oldest first
    interpreted_query: Optional[str]
    standalone_question: Optional[str]  # follow-up rewritten to make sense on its own
    retrieved_chunks: Optional[List[dict]]
    reranked_chunks: Optional[List[dict]]
    retrieval_attempts: int
    date_from: Optional[str]  # ISO date, inclusive upload-date filter (UTC)
    date_to: Optional[str]
    answer: Optional[str]
    citations: Optional[List[dict]]
    trace: List[dict]  # accumulated by each node: {stage, summary, detail}


def route_after_rerank(state: dict) -> str:
    """
    Decides whether to retry retrieval (wider net) or move on to synthesis.
    Retries when the top reranked chunk is below settings.rerank_confidence_threshold
    and we haven't already used up MAX_RETRIEVAL_ATTEMPTS — otherwise
    synthesizes with whatever was found, since an unbounded loop would just
    burn API calls without improving the answer.
    """
    from app.core.config import get_settings

    threshold = get_settings().rerank_confidence_threshold
    reranked = state.get("reranked_chunks") or []
    attempts = state.get("retrieval_attempts", 0)

    top_score = max((c.get("relevance_score", 0) for c in reranked), default=0)

    if not state.get("retrieved_chunks"):
        # Nothing retrieved at all (no documents, or none inside the date filter).
        # A wider top_k cannot add results, so don't spend retries.
        decision = "synthesize"
    elif not reranked or (top_score < threshold and attempts < MAX_RETRIEVAL_ATTEMPTS):
        decision = "retry"
    else:
        decision = "synthesize"
    logger.info(
        "route          -> %s | top_score=%.3f threshold=%.2f attempts=%d/%d",
        decision, top_score, threshold, attempts, MAX_RETRIEVAL_ATTEMPTS,
    )
    return decision


def _logged(name: str, fn):
    """
    Wraps a node so each run logs its timing and the trace entry it produced
    (the same text the UI shows). Failures are logged with the stage name and
    a stack trace, then re-raised unchanged.
    """

    def wrapper(state: dict) -> dict:
        start = time.perf_counter()
        seen = len(state.get("trace", []))
        try:
            out = fn(state)
        except Exception:
            logger.exception("%-14s FAILED after %.0f ms", name, (time.perf_counter() - start) * 1000)
            raise
        ms = (time.perf_counter() - start) * 1000
        for entry in out.get("trace", [])[seen:]:
            logger.info("%-14s %5.0f ms | %s | %s", name, ms, entry["summary"], entry["detail"])
        return out

    return wrapper


def build_graph():
    graph = StateGraph(AgentState)

    from app.agents.query_interpreter import interpret_query_node
    from app.agents.retriever import retrieve_node
    from app.agents.reranker import rerank_node
    from app.agents.synthesizer import synthesize_node

    graph.add_node("interpret_query", _logged("interpret_query", interpret_query_node))
    graph.add_node("retrieve", _logged("retrieve", retrieve_node))
    graph.add_node("rerank", _logged("rerank", rerank_node))
    graph.add_node("synthesize", _logged("synthesize", synthesize_node))

    graph.set_entry_point("interpret_query")
    graph.add_edge("interpret_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {"retry": "retrieve", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    app = graph.compile()
    graph_ascii = app.get_graph().draw_ascii()
    logger.info("Loaded LangGraph Architecture:\n%s", graph_ascii)
    return app
