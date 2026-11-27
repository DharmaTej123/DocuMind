# DocuMind

Multi-agent document intelligence platform for technical documentation
(API references, engineering wikis, PDFs). A LangGraph agent interprets the
question (resolving follow-ups from conversation history), retrieves with hybrid
vector + keyword search, reranks, and writes a grounded, cited answer, or says it
couldn't find one. Evaluation uses RAGAS, with optional LangSmith tracing.

> Status: backend, frontend and evaluation harness are built (see `docs/` for the
> build log). Not yet done: LangSmith analysis, deployment, and the frontend
> changes listed in `docs/FRONTEND.md` ("Backend changes the frontend must adopt").

## What it does
- **Agent graph:** `interpret_query → retrieve → rerank → [retry?] → synthesize`.
  If the best rerank score is below `RERANK_CONFIDENCE_THRESHOLD` (default 0.1),
  retrieval retries once or twice with a wider `top_k` (8 → 14 → 20).
- **Conversation memory:** send the `conversation_id` from a response with your next
  question; follow-ups like "and what tech did he use?" are rewritten into
  self-contained questions. History is stored server-side and scoped to the user.
- **Hybrid retrieval:** vector search fused with BM25 keyword search (reciprocal rank
  fusion), so exact names and numbers match. Switch off with `HYBRID_SEARCH=false`.
- **Guardrails:** retrieved text, questions and history are placed inside delimited
  tags and treated as data; the model replies `NOT_FOUND` when the documents don't
  contain the answer, and you get a plain "couldn't find it" message with no citations.
  Suspicious input is logged. This reduces prompt-injection risk; it does not remove it.
- **PDFs:** page-aware chunking (citations read `p.3`), tables kept intact as Markdown
  (`Table p.3`), and **OCR for scanned pages** via RapidOCR (`OCR p.1`), all local.
- **Document metadata:** one LLM call per upload extracts subject, kind and topics,
  added to every chunk so chunks that say "he" or "this" still carry who or what.
- **Upload-date filter:** `uploaded_after` / `uploaded_before` on `/query/ask`.
- **Pluggable stores and providers:** pgvector (default) or Chroma; Anthropic, OpenAI,
  Gemini or Groq for the LLM.

## Stack
FastAPI · LangGraph · PostgreSQL + pgvector (Chroma optional) · Anthropic (default) /
OpenAI / Gemini / Groq · VoyageAI embeddings (OpenAI and Gemini embeddings when those
providers are selected) · Cohere Rerank · pdfplumber + RapidOCR · RAGAS · LangSmith ·
React/Tailwind · Docker

## Setup

All settings live in one file, **`documind/.env`** (copy from `.env.example`). The
backend finds it by its own location, so it works from any working directory.

```bash
cp .env.example .env
# minimum for the default setup:
#   JWT_SECRET_KEY      (generate: python -c "import secrets; print(secrets.token_urlsafe(48))")
#   ANTHROPIC_API_KEY   (LLM_PROVIDER=anthropic is the default)
#   VOYAGEAI_API_KEY    (embeddings; Anthropic has no embeddings API)
#   COHERE_API_KEY      (reranking)
```

### Docker
```bash
docker compose up --build -d
docker compose exec backend python -m app.db.init_db   # extension, tables, column migrations
```

### Without Docker
Needs Postgres 16 with the pgvector extension. Set `DATABASE_URL` to
`postgresql+psycopg://postgres:postgres@localhost:5432/documind` in `.env`
(the default host `db` only exists inside Docker).
```bash
cd backend
uv sync                      # or: python -m venv .venv && pip install .
python -m app.db.init_db     # run from backend/, as a module
python -m uvicorn app.main:app --port 8000 --reload
cd ../frontend && npm install && npm run dev
```

**Re-run `init_db` after every pull.** It is idempotent and adds columns that
`create_all` would not (`documents.doc_metadata`, `query_logs.conversation_id`).
Skipping it makes document and query requests fail with "column does not exist".

- Health: `GET http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173`

## Using the API
1. `POST /auth/register`, then `POST /auth/login`; authorize with the bearer token.
2. `POST /documents/upload` a `.pdf`, `.md` or `.html` file. Scanned PDFs are OCR'd
   (roughly 5 to 10 seconds per page, so large scans make the request slow).
3. `POST /query/ask`:
   ```json
   {
     "question": "Who built LoopDesk?",
     "conversation_id": null,
     "uploaded_after": "2026-03-01",
     "uploaded_before": "2026-03-31"
   }
   ```
   Omit `conversation_id` to start a conversation; reuse the one returned for
   follow-ups. Dates are optional, inclusive UTC days. The response has `answer`,
   `citations`, `trace` and `conversation_id`.
4. `GET /query/history`, `GET /documents/`, `DELETE /documents/{id}`.

## Configuration
Everything is an environment variable (see `.env.example` for the full list).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `gemini` \| `groq`. A bad value stops the app at startup. |
| `ANTHROPIC_MODEL` etc. | see `.env.example` | Chat model per provider |
| `VECTOR_STORE` | `pgvector` | `pgvector` or `chroma` (temporary local alternative); `CHROMA_PATH` for its data |
| `HYBRID_SEARCH` | `true` | Fuse BM25 keyword search with vector search |
| `RERANK_CONFIDENCE_THRESHOLD` | `0.1` | Retry retrieval below this top rerank score |
| `ANSWER_MIN_RERANK_SCORE` | `0` (off) | Answer "not found" without an LLM call below this score |
| `RERANK_DEBUG` | `true` | **Temporary.** Prints every rerank score and chunk text to the console |
| `HISTORY_TURNS` | `4` | Earlier Q&A pairs shown to the model for follow-ups (0 = off) |
| `EXTRACT_DOC_METADATA` | `true` | One extra LLM call per upload for subject/kind/topics |
| `OCR_ENABLED`, `OCR_DPI`, `OCR_MAX_PAGES` | `true`, `200`, `30` | Scanned-PDF OCR |
| `LOG_LEVEL` | `INFO` | Console log verbosity |

Changing `LLM_PROVIDER`, or `VECTOR_STORE`, means **re-uploading documents**: vectors
from different embedding models aren't comparable, and the two stores aren't migrated.

## Project layout
```
backend/app/
├── api/          # routers: auth, documents, query
├── agents/       # LangGraph graph + nodes; prompting.py holds prompts and guardrails
├── ingestion/    # parser (pdfplumber, OCR), chunker, embedder, metadata, ocr
├── vectorstore/  # pgvector and Chroma backends, hybrid BM25 + fusion, date helpers
├── db/           # SQLAlchemy models, session, init_db (idempotent migrations)
├── eval/         # RAGAS evaluation (collect.py + score.py, see docs/EVAL.md)
└── core/         # config (LLMProvider enum), LLM client, security, logging
frontend/         # React + TypeScript + Tailwind
data/             # sample FastAPI docs used for ingestion testing
docs/             # requirements, build notes, eval, interview guide
```

## Docs
`docs/REQUIREMENTS.md` (scope and decision log) · `docs/EVAL.md` · `docs/FRONTEND.md` ·
`docs/INTERVIEW_GUIDE.md` (architecture walkthrough) · `docs/DATA.md`,
`docs/STEP4_NOTES.md`, `docs/GEMINI_MIGRATION.md` (historical build notes).

## Known limits
- Ingestion is synchronous inside the upload request.
- Retrieval keyword index is built in memory per user (cached 60 seconds); fine for
  thousands of chunks per user, not for very large corpora.
- OCR reads text and rows, not table borders or merged cells, and can misread characters.
- `/query/history` returns questions and answers only, not citations or traces.
