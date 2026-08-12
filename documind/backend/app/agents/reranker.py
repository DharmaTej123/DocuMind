"""
Rerank node — reorders retrieved chunks with Cohere Rerank, which scores
query-chunk relevance far more accurately than raw vector distance (FR6).
Keeps only the top RERANK_KEEP chunks that actually go to synthesis.
"""
from app.core.config import get_settings

RERANK_KEEP = 3


def rerank_node(state: dict) -> dict:
    query = state["interpreted_query"]
    candidates = state["retrieved_chunks"]

    if not candidates:
        trace_entry = {
            "stage": "rerank",
            "summary": "No candidates to rerank.",
            "detail": "retrieved_chunks was empty",
        }
        return {"reranked_chunks": [], "trace": state.get("trace", []) + [trace_entry]}

    import cohere

    settings = get_settings()
    client = cohere.Client(settings.cohere_api_key)

    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[c["content"] for c in candidates],
        top_n=min(RERANK_KEEP, len(candidates)),
    )

    reranked = []
    scores = []
    for result in response.results:
        chunk = candidates[result.index]
        reranked.append({**chunk, "relevance_score": result.relevance_score})
        scores.append(round(result.relevance_score, 3))

    trace_entry = {
        "stage": "rerank",
        "summary": f"Reranked {len(candidates)} candidates, kept top {len(reranked)}.",
        "detail": f"scores={scores}, dropped={len(candidates) - len(reranked)}",
    }

    return {
        "reranked_chunks": reranked,
        "trace": state.get("trace", []) + [trace_entry],
    }
