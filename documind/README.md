# DocuMind

Multi-agent document intelligence platform for technical documentation
(API references, engineering wikis). LangGraph-orchestrated agents handle
query interpretation, retrieval, reranking, and grounded answer synthesis,
with RAGAS + LangSmith evaluation on every query.

> Status: Steps 1-7 built (requirements through evaluation harness). See
> docs/ for the full build log — REQUIREMENTS.md, DATA.md, STEP4_NOTES.md,
> EVAL.md, GEMINI_MIGRATION.md. Steps 8-12 (LangSmith analysis,
> containerization verification, deployment, polished README, GitHub) are
> still open.

## Stack
FastAPI · LangGraph · pgvector · Gemini (default) / OpenAI · Cohere Rerank · RAGAS · LangSmith · React/Tailwind · Docker · Railway

## Local dev
```bash
cp .env.example .env
# fill in: GEMINI_API_KEY, COHERE_API_KEY, JWT_SECRET_KEY
docker compose up --build -d
docker compose exec backend python -m app.db.init_db   # creates pgvector extension + tables
```
Backend health check: `GET http://localhost:8000/health`
Frontend: `http://localhost:5173`
API docs (Swagger UI): `http://localhost:8000/docs`

`init_db` is idempotent — safe to re-run any time (e.g. after pulling schema
changes) without dropping data.

See `docs/RUNNING_LOCALLY.md` for the manual (non-Docker) setup path and
troubleshooting notes.

## Project layout
```
backend/app/
├── api/         # FastAPI routers: auth, documents, query
├── agents/      # LangGraph nodes + graph definition (with conditional retry)
├── ingestion/   # parsing, chunking, embedding
├── db/          # SQLAlchemy models, pgvector, session
├── eval/        # RAGAS evaluation (collect.py + score.py, see docs/EVAL.md)
└── core/        # config, auth, LLM client abstraction
frontend/        # React + TypeScript + Tailwind
data/            # sample FastAPI docs used for ingestion testing
```
