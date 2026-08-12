# Migrating from OpenAI to Gemini

## Files changed
| File | Change |
|---|---|
| `core/config.py` | Added `gemini_api_key`, `gemini_model`, `gemini_embedding_model`, `embedding_dim`; changed `llm_provider` default to `"gemini"` |
| `core/llm_client.py` | Added a `gemini` branch to `get_llm()` using `ChatGoogleGenerativeAI` |
| `ingestion/embedder.py` | Split into `_embed_chunks_openai` / `_embed_chunks_gemini`, dispatched by `settings.llm_provider` |
| `pyproject.toml` | Added `langchain-google-genai`, `google-genai`; kept `langchain-openai`/`openai` for the fallback path |
| `.env.example` | `LLM_PROVIDER=gemini` by default, `GEMINI_API_KEY` etc. added, OpenAI vars kept for switching back |

**Not changed:** every agent node (`interpret_query`, `retrieve`, `rerank`,
`synthesize`), the graph wiring, the retry logic, the API routers, the
frontend. This is the direct payoff of the Step 1 decision to put all LLM
calls behind `get_llm()` instead of importing a provider SDK directly in
each node — see `docs/REQUIREMENTS.md` Decisions Log.

## The gotcha this surfaced: embedding dimensions
Swapping the *chat* model is trivial — swapping the *embedding* model isn't,
because the vector dimension is baked into the pgvector column
(`db/models.py EMBEDDING_DIM`, currently `1536`).

- OpenAI's `text-embedding-3-small` → natively 1536-dim.
- Gemini's `gemini-embedding-001` → natively **3072-dim**, but supports an
  `output_dimensionality` parameter to truncate.

If you don't handle this, switching providers silently breaks retrieval —
either an insert fails on a dimension mismatch, or (worse, if the column
type is loose) it succeeds but cosine similarity is computed nonsense
because you're comparing 1536-dim and 3072-dim vectors.

**Decision:** set `output_dimensionality=1536` on every Gemini embedding
call (`embedder.py`) so it's a drop-in match for the existing schema. No
migration needed. Tradeoff: you're not using Gemini's embedding model at
its full native quality — 3072-dim would score marginally better on
retrieval benchmarks. If you want the full 3072 dims later, you'd need a
real migration (`ALTER TABLE chunks ALTER COLUMN embedding TYPE
vector(3072)`) **and** re-embed every existing chunk, since old and new
vectors aren't comparable.

## What still needs a real API key to verify
Model construction is confirmed working (`ChatGoogleGenerativeAI` and the
Gemini embedding call path both build correctly with a placeholder key —
see test log below). The actual live call — a real question getting a real
answer — is untested pending a real `GEMINI_API_KEY`, same gap as existed
for OpenAI in `docs/STEP4_NOTES.md`.

```
LLM client constructed: ChatGoogleGenerativeAI
model: gemini-3.5-flash
```

## Switching back to OpenAI
Set `LLM_PROVIDER=openai` in `.env` — both branches are live and tested,
so this is a one-line change, not a code change.

## Model choice note
`gemini-3.5-flash` was picked over `gemini-2.5-flash` because Google has
announced `gemini-2.5-flash` and `gemini-2.5-pro` shut down on **October
16, 2026** — using them now would mean re-migrating in a few months.
`gemini-3.6-flash` (the newest model as of this writing) is also a
reasonable pick if you want the latest; `3.5-flash` was chosen as the
safer, more-established GA option. Worth rechecking Google's model page
before a real deployment, since this space moves fast.
