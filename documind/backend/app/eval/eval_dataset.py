"""
Fixed evaluation set — NFR5 (reproducible eval).

Each entry has a ground-truth reference answer, written by hand from the
actual source docs (not generated), so RAGAS's context_recall and
answer-quality metrics have something real to compare against. Spans the
four question types identified in docs/DATA.md: direct lookup, cross-doc,
code-dependent, and multi-step.
"""

EVAL_SET = [
    {
    "id": "q001",
    "question": "What are the four core RAGAS metrics listed in the DocuMind study plan?",
    "reference": "Faithfulness, answer relevancy, context precision, and context recall.",
    "category": "factual"
  },
  {
    "id": "q002",
    "question": "What does the study plan say faithfulness measures?",
    "reference": "It measures whether the generated answer is grounded in the retrieved context.",
    "category": "factual"
  },
  {
    "id": "q003",
    "question": "What does answer relevancy measure in RAGAS?",
    "reference": "It measures how relevant the generated answer is to the user's question.",
    "category": "factual"
  },
  {
    "id": "q005",
    "question": "What does context recall measure?",
    "reference": "It measures whether the retrieval process finds the information needed to answer the question.",
    "category": "factual"
  },
]