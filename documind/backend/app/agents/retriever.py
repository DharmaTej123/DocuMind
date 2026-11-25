"""
Retrieval node — embeds the interpreted query and pulls the top-K nearest
chunks from the active vector store (pgvector by default, or chroma),
scoped to the current user's own documents only.

On a retry (state["retrieval_attempts"] > 0, routed back here by
route_after_rerank in graph.py because the last pass scored low
confidence), top_k widens so the retry actually casts a wider net instead
of repeating the same search.
"""
from app import vectorstore
from app.ingestion.embedder import embed_query

BASE_TOP_K = 8
WIDEN_PER_ATTEMPT = 6


def retrieve_node(state: dict) -> dict:
    query = state["interpreted_query"]
    user_id = state["user_id"]
    attempts = state.get("retrieval_attempts", 0)
    top_k = BASE_TOP_K + (attempts * WIDEN_PER_ATTEMPT)

    query_embedding = embed_query(query)

    # Keyword side sees the user's original wording too: the rewritten query
    # may drop exact names or figures that BM25 should match.
    date_from, date_to = state.get("date_from"), state.get("date_to")
    retrieved = vectorstore.search(
        query_embedding, user_id, top_k,
        query_text=f"{state.get('standalone_question') or state['question']} {query}",
        date_from=date_from, date_to=date_to,
    )

    via = [r.get("retrieved_via") for r in retrieved]
    mode = (
        f"hybrid (vector-only={via.count('vector')}, keyword-only={via.count('keyword')}, "
        f"both={via.count('both')})"
        if any(via)
        else "vector"
    )
    retry_note = f" (retry #{attempts}, widened top_k)" if attempts else ""
    trace_entry = {
        "stage": "retrieve",
        "summary": f"Pulled {len(retrieved)} candidate chunks from {vectorstore.active_store()}, {mode}.{retry_note}",
        "detail": f"top_k={top_k}, sources={sorted({r['doc'] for r in retrieved})}"
        + (f", uploaded {date_from or '...'} to {date_to or '...'}" if date_from or date_to else ""),
    }

    return {
        "retrieved_chunks": retrieved,
        "retrieval_attempts": attempts + 1,
        "trace": state.get("trace", []) + [trace_entry],
    }
