"""
Query interpretation node — rewrites the user's raw question into a
retrieval-friendly query (expands abbreviations, strips conversational
filler) before it hits the vector store.
"""
import logging

from app.core.llm_client import get_llm

SYSTEM_PROMPT = (
    "You rewrite user questions into short, keyword-rich search queries for "
    "a technical documentation retrieval system. Return only the rewritten "
    "query, nothing else."
)


def interpret_query_node(state: dict) -> dict:
    question = state["question"]
    llm = get_llm()

    result = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    )
    logging.info(f"LLM output: {result.content}")
    interpreted = result.content.strip()

    trace_entry = {
        "stage": "interpret",
        "summary": f"Rewrote question into a retrieval query: \"{interpreted}\"",
        "detail": f"original=\"{question}\"",
    }

    return {
        "interpreted_query": interpreted,
        "trace": state.get("trace", []) + [trace_entry],
    }
