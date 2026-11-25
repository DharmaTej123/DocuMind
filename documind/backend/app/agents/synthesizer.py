"""
Synthesis node — generates the final grounded answer from the reranked
chunks, with inline citations back to source doc + section (FR5).
"""
from app.core.llm_client import get_llm
from app.core.config import get_settings

SYSTEM_PROMPT = (
    "You answer technical documentation questions using ONLY the provided "
    "context chunks. If the context doesn't contain the answer, say so "
    "plainly instead of guessing. Be concise and precise — this is API "
    "reference material, not prose to pad."
)


def synthesize_node(state: dict) -> dict:
    question = state["question"]
    chunks = state["reranked_chunks"]

    context = "\n\n---\n\n".join(
        f"[Source: {c['doc']} § {c['section']}]\n{c['content']}" for c in chunks
    )

    llm = get_llm()
    result = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
    )
    answer = result.content.strip()

    citations = [{"doc": c["doc"], "section": c["section"]} for c in chunks]

    settings = get_settings()
    trace_entry = {
        "stage": "synthesize",
        "summary": "Answer generated and grounded in the reranked chunks.",
        "detail": f"model={getattr(settings, f'{settings.llm_provider}_model', settings.llm_provider)}, citations_attached={len(citations)}",
    }

    return {
        "answer": answer,
        "citations": citations,
        "trace": state.get("trace", []) + [trace_entry],
    }
