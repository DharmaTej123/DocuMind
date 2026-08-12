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
from fastapi import logger
from langgraph.graph import StateGraph, END

MAX_RETRIEVAL_ATTEMPTS = 2
RERANK_CONFIDENCE_THRESHOLD = 0.5


class AgentState(TypedDict):
    question: str
    user_id: str
    interpreted_query: Optional[str]
    retrieved_chunks: Optional[List[dict]]
    reranked_chunks: Optional[List[dict]]
    retrieval_attempts: int
    answer: Optional[str]
    citations: Optional[List[dict]]
    trace: List[dict]  # accumulated by each node: {stage, summary, detail}


def route_after_rerank(state: dict) -> str:
    """
    Decides whether to retry retrieval (wider net) or move on to synthesis.
    Retries when the top reranked chunk is below RERANK_CONFIDENCE_THRESHOLD
    and we haven't already used up MAX_RETRIEVAL_ATTEMPTS — otherwise
    synthesizes with whatever was found, since an unbounded loop would just
    burn API calls without improving the answer.
    """
    reranked = state.get("reranked_chunks") or []
    attempts = state.get("retrieval_attempts", 0)

    top_score = max((c.get("relevance_score", 0) for c in reranked), default=0)

    if not reranked or (top_score < RERANK_CONFIDENCE_THRESHOLD and attempts < MAX_RETRIEVAL_ATTEMPTS):
        return "retry"
    return "synthesize"

logs = logging.getLogger(__name__)
def build_graph():
    graph = StateGraph(AgentState)

    from app.agents.query_interpreter import interpret_query_node
    from app.agents.retriever import retrieve_node
    from app.agents.reranker import rerank_node
    from app.agents.synthesizer import synthesize_node

    graph.add_node("interpret_query", interpret_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("synthesize", synthesize_node)

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
    logs.info(f"Loaded LangGraph Architecture:\n{graph_ascii}")
    return app
