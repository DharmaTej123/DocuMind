"""
Phase 2 of evaluation — runs in the SEPARATE RAGAS-only venv (see
docs/EVAL.md for setup). Reads the plain-JSON output of collect.py and
scores it with RAGAS: faithfulness, answer relevancy, context precision,
context recall.

Judge model: Groq (llama-3.3-70b-versatile), not OpenAI. RAGAS's evaluate()
defaults to an OpenAI judge if you don't pass one explicitly — this script
always passes an explicit LangchainLLMWrapper around ChatGroq instead.

answer_relevancy is the one metric that also needs an embeddings model
(it embeds a reverse-generated question and compares it to the original).
Groq has no embeddings API, so this reuses Gemini's embedding model — same
provider the app already uses for retrieval — rather than pulling in a
third provider just for this one metric.

Requires GROQ_API_KEY and GEMINI_API_KEY in this venv's environment.

Deliberately has ZERO imports from the `app` package — this script only
touches the JSON file collect.py produced, so it never needs langgraph or
any of the app's dependencies, which is what makes running it in a
different venv safe.

Usage:
    python score.py --in eval_raw.json --out eval_scored.json
"""
import argparse
import json
import os
from datetime import datetime, timezone

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"


def load_rows(in_path: str) -> list[dict]:
    with open(in_path) as f:
        payload = json.load(f)
    return payload["rows"]


def build_ragas_dataset(rows: list[dict]):
    from ragas import EvaluationDataset

    samples = [
        {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row["contexts"] or [""],  # RAGAS requires non-empty
            "reference": row["reference"],
        }
        for row in rows
    ]
    return EvaluationDataset.from_list(samples)


def build_judge():
    """Groq LLM wrapped for RAGAS, used by all four metrics."""
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set in this venv's environment")

    return LangchainLLMWrapper(ChatGroq(model=GROQ_MODEL, api_key=groq_key, temperature=0))


def build_judge_embeddings():
    """Gemini embeddings wrapped for RAGAS, used only by answer_relevancy."""
    import voyageai
    from ragas.embeddings import LangchainEmbeddingsWrapper

    voyageai_key = os.environ.get("VOYAGEAI_API_KEY")
    if not voyageai_key:
        raise RuntimeError("VOYAGEAI_API_KEY not set in this venv's environment")

    return LangchainEmbeddingsWrapper(
        voyageai.Client(api_key=voyageai_key)
    )
    


def score(in_path: str, out_path: str):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    rows = load_rows(in_path)
    dataset = build_ragas_dataset(rows)

    judge_llm = build_judge()
    judge_embeddings = build_judge_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    df = result.to_pandas()

    per_question = []
    for i, row in enumerate(rows):
        per_question.append(
            {
                "question": row["question"],
                "type": row["type"],
                "faithfulness": float(df.iloc[i]["faithfulness"]),
                "answer_relevancy": float(df.iloc[i]["answer_relevancy"]),
                "context_precision": float(df.iloc[i]["context_precision"]),
                "context_recall": float(df.iloc[i]["context_recall"]),
            }
        )

    aggregate = {
        metric: float(df[metric].mean())
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    }

    payload = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(rows),
        "aggregate_scores": aggregate,
        "per_question": per_question,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Scored {len(rows)} questions.")
    print(json.dumps(aggregate, indent=2))
    print(f"Full results -> {out_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="eval_raw.json")
    parser.add_argument("--out", default="eval_scored.json")
    args = parser.parse_args()
    score(in_path=args.in_path, out_path=args.out)
