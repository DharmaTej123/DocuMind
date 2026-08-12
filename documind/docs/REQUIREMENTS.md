# DocuMind — Requirements

## 1. Problem Statement
Engineers spend significant time searching scattered technical documentation (API
references, engineering wikis, internal runbooks) for answers that are often buried
in outdated, duplicated, or poorly indexed pages. Naive RAG (embed → retrieve → generate)
degrades on technical docs because:
- Similar API methods/classes produce near-duplicate embeddings, so vector search alone
  retrieves plausible-but-wrong chunks.
- Multi-step questions ("how do I do X, and what are the rate limits for that endpoint")
  need more than one retrieval pass.
- There's no way to know if an answer is trustworthy without a human re-checking the source.

DocuMind addresses this with a multi-agent architecture: dedicated agents for query
understanding, retrieval, and synthesis, a rerank stage to fix retrieval precision, and an
automated evaluation loop (RAGAS + LangSmith) that scores faithfulness and relevance on
every answer.

## 2. Domain & Scope
- **Document domain:** technical documentation — API references, engineering wiki pages,
  README/runbook-style docs (Markdown, HTML, PDF).
- **MVP user model:** multi-user with authentication. Each user has their own workspace
  (uploaded docs + query history); documents are not shared across users in MVP.
- **Out of scope for MVP:** team/org-level sharing, fine-grained permissions, real-time
  collaborative editing, non-English docs.

## 3. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | User can register/log in (email + password, JWT session) |
| FR2 | User can upload one or more documents (PDF, Markdown, HTML) to their workspace |
| FR3 | System ingests documents: parses, chunks, embeds, stores in pgvector |
| FR4 | User can ask a natural-language question about their uploaded docs |
| FR5 | System returns an answer grounded in retrieved chunks, with source citations (doc name + section) |
| FR6 | System reranks retrieved chunks (Cohere Rerank) before answer synthesis |
| FR7 | Multi-agent graph (LangGraph) handles: query interpretation → retrieval → (optional re-query) → synthesis |
| FR8 | Every query is evaluated (RAGAS: faithfulness, answer relevance, context precision) and logged to LangSmith |
| FR9 | User can view past queries and answers (history) in their workspace |
| FR10 | User can delete an uploaded document (and its vectors) |

## 4. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | API responses for a query should return in <10s for MVP (no hard SLA — measured, not optimized, first) |
| NFR2 | System runs fully containerized (Docker Compose for local dev) |
| NFR3 | Secrets (API keys, DB creds) via environment variables, never committed |
| NFR4 | Codebase is cloud-agnostic at the app layer (Postgres/pgvector, not a proprietary vector DB) |
| NFR5 | Evaluation metrics must be reproducible — a fixed eval set of Q&A pairs re-run on demand |

## 5. Tech Stack

| Layer | Choice |
|-------|--------|
| Backend API | FastAPI |
| Agent orchestration | LangGraph |
| LLM | Claude / OpenAI (pluggable) |
| Vector store | pgvector (Postgres) |
| Reranking | Cohere Rerank |
| Evaluation | RAGAS |
| Observability/tracing | LangSmith |
| Frontend | React + shadcn/ui |
| Auth | JWT (FastAPI + passlib) |
| Containerization | Docker, Docker Compose |
| Deployment | Railway |

## 6. Success Criteria (for interview narrative)
- End-to-end demo: upload docs → ask question → grounded answer with citations
- A documented before/after: naive single-pass RAG vs. reranked multi-agent RAG, with
  RAGAS scores showing the improvement
- Deployed, publicly reachable instance + GitHub repo with README

## 7. Open Questions (to resolve during build)
- Chunking strategy specifics (fixed-size vs. structure-aware for Markdown/HTML)
- Whether query rewriting/decomposition is a separate agent node or folded into retrieval

## 8. Decisions Log
- **LLM provider:** ~~OpenAI~~ **Gemini** (`gemini-3.5-flash`), switched during
  build. Kept pluggable per NFR-style design (agent nodes call
  `core/llm_client.get_llm()`, never a provider SDK directly) — this paid
  off exactly as intended: the switch was a ~15-line diff in
  `llm_client.py` and `embedder.py`, not a rewrite. See
  `docs/GEMINI_MIGRATION.md` for the full story, including the embedding
  dimension gotcha this switch surfaced.
