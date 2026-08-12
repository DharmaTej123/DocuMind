# Step 4 — Agents & Ingestion: Build Notes

## What's implemented
- **Auth** (`core/security.py`, `core/deps.py`, `api/auth.py`): bcrypt password
  hashing, JWT issue/verify, `get_current_user` dependency.
- **Ingestion** (`ingestion/parser.py`, `chunker.py`, `embedder.py`): parses
  PDF/Markdown/HTML, resolves the FastAPI docs' `{* *}` code-snippet macros
  (per `docs/DATA.md`), structure-aware chunking by headers, OpenAI embeddings.
- **Agents** (`agents/*.py`): `interpret_query` (LLM query rewrite) →
  `retrieve` (pgvector cosine search, scoped to the user's own docs) →
  `rerank` (Cohere Rerank, top 3 kept) → `synthesize` (grounded answer +
  citations). Every node appends a `trace` entry matching the frontend's
  `AgentTrace` component contract.
- **API** (`api/documents.py`, `api/query.py`): upload triggers the full
  ingestion pipeline synchronously; `/query/ask` invokes the compiled graph.

## What's verified (real tests run in this environment)
- **Parser**: all 24 sample docs resolve their code macros with zero
  failures (caught and fixed a path-prefix bug).
- **Chunker**: 347 chunks across the sample set, zero oversized — caught and
  fixed a real bug where `#`-prefixed comment lines *inside fenced code
  blocks* (e.g. bash `# comment`) were misdetected as Markdown headers.
- **Auth**: password hash/verify and JWT create/decode round-tripped
  correctly, including negative cases (wrong password, tampered token).
- **Full stack, live**: installed Postgres 16 + pgvector locally, created all
  4 tables, ran the actual FastAPI server, and confirmed:
  - register → real user row created
  - login → real JWT issued; wrong password correctly rejected (401)
  - protected routes correctly reject missing/invalid tokens (401)
  - `/documents/upload` correctly reaches the embedding call before failing
    (see below)
  - `/query/ask` correctly invokes the LangGraph pipeline and reaches
    `interpret_query`'s LLM call before failing (see below)

## Known gap — needs real API keys to finish verifying
This sandbox has no `OPENAI_API_KEY` or `COHERE_API_KEY`. Everything up to
the first external API call is confirmed working; the calls themselves
(`embed_chunks`, `interpret_query_node`'s `llm.invoke`, `rerank_node`'s
Cohere call, `synthesize_node`'s `llm.invoke`) are implemented and reviewed
but not live-tested. To finish Step 4 verification:
```bash
# in backend/.env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```
then re-run `POST /documents/upload` with a sample file and `POST /query/ask`
with a question — both should return real, grounded answers.

## Notable decisions made during implementation
- Ingestion runs **synchronously** on upload (parse→chunk→embed→store in one
  request). Fine for MVP file sizes; flagged as a background-queue upgrade if
  upload latency becomes a problem later.
- `QueryLog` currently stores only question/answer, not citations/trace —
  so `/query/history` replays answers but with empty citations/trace. Worth
  extending if the history view should show the full trace on replay.
