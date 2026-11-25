"""
Rerank node — reorders retrieved chunks with Cohere Rerank, which scores
query-chunk relevance far more accurately than raw vector distance (FR6).
Keeps only the top RERANK_KEEP chunks that actually go to synthesis.

The reranker is given the user's ORIGINAL question, not the keyword rewrite
used for retrieval: Cohere's reranker is trained on natural-language queries
and scores keyword strings lower.
"""
from app.core.config import get_settings

RERANK_KEEP = 3


def rerank_node(state: dict) -> dict:
    # The reranker can't see conversation history, so use the self-contained
    # version of a follow-up ("and what tech did he use?" -> full question).
    query = state.get("standalone_question") or state["question"]
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

    # With rerank_debug on, ask Cohere to score EVERY candidate (same single
    # API call) so the full ranking can be inspected; we still keep RERANK_KEEP.
    top_n = len(candidates) if settings.rerank_debug else min(RERANK_KEEP, len(candidates))
    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[c["content"] for c in candidates],
        top_n=top_n,
    )

    # --- TEMPORARY DEBUG (remove, or set RERANK_DEBUG=false) ---------------
    if settings.rerank_debug:
        print(f"\n===== RERANK DEBUG | query sent to Cohere: {query!r} | candidates={len(candidates)} =====")
        for position, result in enumerate(response.results, start=1):
            candidate = candidates[result.index]
            kept = "KEPT   " if position <= RERANK_KEEP else "dropped"
            print(f"\n#{position} [{kept}] Rank(index in retrieved list): {result.index}")
            print("Score:", repr(result.relevance_score))
            print("Source:", candidate["doc"], "|", candidate.get("section"), "| via:", candidate.get("retrieved_via", "vector"))
            print("Document:", candidate["content"][:500])
        print("===== END RERANK DEBUG =====\n", flush=True)
    # -----------------------------------------------------------------------

    reranked = []
    scores = []
    for result in response.results[:RERANK_KEEP]:
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
