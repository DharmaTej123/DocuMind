"""
Phase 1 of evaluation — runs in the MAIN app venv (has langgraph, the real
agent graph, DB access). Invokes the graph for every question in
eval_dataset.py and writes plain-JSON results (no LangChain objects) to
disk. Phase 2 (score.py) reads that file in a separate, RAGAS-only venv.

Why two phases / two venvs: RAGAS's import chain pulls an old
langchain-community that requires langchain-core <1, while langgraph/
langchain-openai (used by the actual agent graph) require langchain-core
>=1.4. Both can't be installed in one environment right now — see
docs/EVAL.md for the full story. Splitting generation from scoring sidesteps
the conflict entirely and is also just a cleaner separation of concerns:
"produce answers" and "grade answers" don't need to share a dependency tree.

Usage:
    python -m app.eval.collect --user-id <uuid> --out eval_raw.json
"""
import argparse
import json
from datetime import datetime, timezone

from app.eval.eval_dataset import EVAL_SET
from app.agents.graph import build_graph


def collect(user_id: str, out_path: str) -> list[dict]:
    graph = build_graph()
    rows = []

    for item in EVAL_SET:
        result = graph.invoke(
            {"question": item["question"], "user_id": user_id, "trace": [], "retrieval_attempts": 0}
        )
        rows.append(
            {
                "question": item["question"],
                "category": item["category"],
                "reference": item["reference"],
                "answer": result["answer"],
                "contexts": [c["content"] for c in result.get("reranked_chunks", [])],
                "citations": result.get("citations", []),
                "trace": result.get("trace", []),
            }
        )

    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(rows),
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"Collected {len(rows)} answers -> {out_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True, help="User ID whose documents to query against")
    parser.add_argument("--out", default="eval_raw.json")
    args = parser.parse_args()
    collect(user_id=args.user_id, out_path=args.out)
