# Evaluation — Step 7

> **Status update (current code vs. the notes below)**
> - **Fixed:** `score.py` read `row["type"]` but `collect.py` writes `category`, which
>   would have raised `KeyError` after scoring. It now reads `category`.
> - **Judge embeddings are VoyageAI, not Gemini.** `score.py` builds its `answer_relevancy`
>   embeddings from `voyageai.Client` (needs `VOYAGEAI_API_KEY`, not `GEMINI_API_KEY`), and
>   its docstring still says Gemini. The "Gemini" paragraphs below are out of date. The code
>   passes the raw `voyageai.Client` to `LangchainEmbeddingsWrapper`, which expects a
>   LangChain embeddings object, so that path likely needs a proper wrapper. It has not
>   been run.
> - **The eval set has 4 questions** (about the RAGAS study plan; `q004` is missing), not
>   the 10 described under "Eval set". Too small for more than a regression check.
> - **What `collect.py` exercises now:** the real graph with whatever `LLM_PROVIDER`,
>   `VECTOR_STORE` and `HYBRID_SEARCH` are active, so record those with each run. It sends
>   no conversation history (single-turn). Set `RERANK_DEBUG=false` for a clean console.
> - **Score caveats:** contexts passed to RAGAS are the reranked chunks, which now start with
>   a `[file | kind | subject | topics | section]` prefix. A question the documents can't
>   answer now yields the "couldn't find the answer" message, which RAGAS may score low on
>   `answer_relevancy`; that's the system declining, not a retrieval failure.

## Architecture: two phases, two venvs
Evaluation is split into **collect** (generate answers) and **score** (grade
them with RAGAS), run in two separate Python environments:

```
backend/app/eval/
├── eval_dataset.py   # 10 fixed Q&A pairs, hand-written references
├── collect.py        # Phase 1 — runs in the MAIN app venv
├── score.py           # Phase 2 — runs in a SEPARATE ragas-only venv
└── requirements-eval.txt
```

```bash
# Phase 1 — main app venv (has langgraph, DB access, real graph)
cd backend
python -m app.eval.collect --user-id <uuid> --out eval_raw.json

# Phase 2 — separate venv (has RAGAS)
python3 -m venv ../eval_venv
../eval_venv/bin/pip install -r app/eval/requirements-eval.txt
../eval_venv/bin/python app/eval/score.py --in eval_raw.json --out eval_scored.json
```

## Judge model: Groq + Gemini, not OpenAI
`score.py` passes an explicit judge LLM and embeddings model to
`evaluate()` — it does not rely on RAGAS's OpenAI default:
- **LLM judge** (all 4 metrics): Groq, `llama-3.3-70b-versatile`, via
  `LangchainLLMWrapper(ChatGroq(...))`
- **Embeddings** (`answer_relevancy` only — it embeds a reverse-generated
  question and compares it to the original): Gemini's
  `gemini-embedding-001`, via `LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(...))`
  — reusing the same provider the app already uses for retrieval rather
  than adding a third API key just for one metric.

Requires `GROQ_API_KEY` and `GEMINI_API_KEY` set in the **eval venv's**
environment (separate from the main app's `.env` — see setup below).

Adding `langchain-groq` surfaced the *same class* of dependency conflict
as the RAGAS/vertexai issue below: its latest version pulls a newer
`langchain-core` that breaks the pinned stack. Fix was the same shape —
pin `langchain-groq==0.2.5` (needs `langchain-core>=0.3.42,<0.4`) and
`langchain-google-genai==2.1.0` (needs `langchain-core>=0.3.43,<0.4`),
then widen the core pin to `>=0.3.43,<0.4` to satisfy both instead of the
original exact `==0.3.29` pin. Verified fresh-install clean with the exact
`requirements-eval.txt` in this repo (no pip conflict warnings).

## Why two venvs — a real, current dependency conflict
This isn't over-engineering — it's a genuine incompatibility as of the
package versions available today (Aug 2026):

- RAGAS (both 0.2.15 and the latest 0.4.3) unconditionally imports
  `ChatVertexAI` from `langchain_community.chat_models.vertexai` at import
  time, even if you never touch Vertex AI.
- Recent `langchain-community` releases (it's now in maintenance/sunset
  mode) have dropped that module as part of migrating integrations to
  standalone packages.
- RAGAS's own package doesn't pin an upper bound on `langchain-community`,
  so a plain `pip install ragas` pulls the newest one and breaks on import
  with `ModuleNotFoundError: No module named
  'langchain_community.chat_models.vertexai'`.
- Fixing it requires pinning `langchain-community` back to ~0.3.14, which
  drags `langchain-core` down to the 0.3.x line — but `langgraph` and
  `langchain-openai` (used by the actual agent graph) require
  `langchain-core>=1.4`. Both constraints can't be satisfied in one
  environment.

Verified this hands-on: reproduced the failure, tried three different pin
combinations before finding the coherent old-stack combo that actually
imports (`ragas==0.2.15` + `langchain==0.3.14` + `langchain-core>=0.3.43,<0.4`
+ `langchain-openai==0.2.14` + `langchain-community==0.3.14` +
`langchain-groq==0.2.5` + `langchain-google-genai==2.1.0`), confirmed with
real imports of `evaluate`, `EvaluationDataset`, all four metrics, `ChatGroq`,
and `GoogleGenerativeAIEmbeddings` together in one venv.

Splitting collect/score across venvs sidesteps this cleanly, and is
arguably better design anyway — "generate an answer" and "grade an answer"
are genuinely different concerns with no reason to share a dependency tree.
`score.py` has zero imports from the `app` package on purpose, so it never
needs to see langgraph at all.

## Eval set
10 fixed questions in `eval_dataset.py`, spanning the four types identified
during Step 3 data collection: direct lookup, cross-doc, code-dependent,
multi-step. Each has a hand-written reference answer (not LLM-generated),
so `context_recall` has a real ground truth to check retrieval against.

## Metrics
- **faithfulness** — is the answer actually supported by the retrieved context?
- **answer_relevancy** — does the answer address the question asked?
- **context_precision** — are the retrieved chunks actually relevant?
- **context_recall** — did retrieval surface what the reference answer needed?

## Verified in this environment
- `collect.py`'s graph invocation logic reuses the same `build_graph()`
  wiring already tested in Step 4/5 (real Postgres, real conditional retry).
- `score.py`'s `build_ragas_dataset()` and `load_rows()` — tested against a
  fake `eval_raw.json` — build a valid RAGAS `EvaluationDataset` correctly.
- `build_judge()` and `build_judge_embeddings()` construct real
  `ChatGroq`/`GoogleGenerativeAIEmbeddings` instances wrapped correctly for
  RAGAS, confirmed with placeholder API keys.
- Fresh `pip install -r requirements-eval.txt` in a clean venv installs
  with no dependency-resolver conflicts.

## Not yet verified — needs real API keys
The actual metric scores require real `GROQ_API_KEY` and `GEMINI_API_KEY`
values set in the eval venv's environment (not the main app's `.env` —
these are separate environments by design, see above). Everything up to
the actual scoring call is confirmed working; run both phases with real
keys for real numbers.
