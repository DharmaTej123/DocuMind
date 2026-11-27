# DocuMind — AI Engineer Interview Guide

> A complete reference for explaining the DocuMind project in an interview.
> Contains: (1) a 30-second pitch, (2) an interview-style narrative, (3) an
> extended technical deep-dive, (4) a one-page cheat sheet, (5) the agent-graph
> diagram, and (6) mock Q&A with likely follow-ups.

---

## Table of Contents

1. [30-Second Pitch](#1-30-second-pitch)
2. [Interview-Style Explanation](#2-interview-style-explanation)
3. [Extended Technical Deep-Dive](#3-extended-technical-deep-dive)
4. [One-Page Cheat Sheet](#4-one-page-cheat-sheet)
5. [Agent Graph Diagram](#5-agent-graph-diagram)
6. [Mock Interview Q&A](#6-mock-interview-qa)
7. [Known Gaps / "What I'd Improve"](#7-known-gaps--what-id-improve)
8. [RAGAS Metrics — How to Interpret Scores](#8-ragas-metrics--how-to-interpret-scores)
9. [Live Demo Script](#9-live-demo-script)
10. [Recent Upgrades — Talking Points and Q&A](#10-recent-upgrades--talking-points-and-qa)

---

## 1. 30-Second Pitch

DocuMind is a **multi-tenant RAG application for Q&A over technical documentation**.
Users upload docs (PDF / Markdown / HTML); the system parses, chunks, embeds, and
stores them; then they ask natural-language questions.

The differentiator is the answer engine: it's not a linear RAG chain, it's an
**agentic pipeline built on LangGraph** that interprets the query, does two-stage
retrieval (vector search + cross-encoder rerank), and **conditionally retries
retrieval with a wider net** when confidence is low — before synthesizing a
grounded, cited answer. Every step emits a trace the UI renders, so the pipeline
is fully transparent.

**Architecture at a glance:**

![DocuMind architecture overview](images/architecture-overview.svg)

**One-sentence opener:**
> "DocuMind is a multi-tenant RAG system over technical docs, but the piece I'm
> proudest of is that the answer engine is a real LangGraph agent — it interprets
> the query, does two-stage retrieval, and loops back to retry retrieval with a
> wider net when the reranker's confidence is low, all bounded and traced."

---

## 2. Interview-Style Explanation

### The problem

Technical documentation is painful to search. Keyword search misses paraphrases,
and dumping a whole doc into an LLM is expensive and imprecise. DocuMind lets a
user upload their own docs and ask questions in natural language, getting back a
grounded answer with citations back to the exact source section.

### How I built it (prototype → finished)

I'd tell this as a phased story, because it shows engineering judgment:

1. **Skeleton & contracts** — FastAPI app, three routers (`/auth`, `/documents`,
   `/query`), Pydantic schemas kept separate from the DB models, Postgres +
   pgvector as the datastore, JWT auth from day one because it's multi-tenant.
2. **Ingestion pipeline** — parse → chunk → embed → store, run synchronously in
   the upload endpoint (fine for MVP file sizes).
3. **Baseline RAG** — vector-only cosine retrieval straight into the LLM.
4. **Make it agentic** — wrapped the flow in a LangGraph state machine, added a
   query-interpretation node, a Cohere rerank stage, and a conditional retrieval
   retry loop. *This is the heart of the project.*
5. **Grounding & transparency** — strict grounding prompt, inline citations,
   per-node execution trace returned to the UI.
6. **Evaluation** — a reproducible RAGAS harness (faithfulness, answer relevancy,
   context precision, context recall), split into two phases to dodge a
   dependency conflict.
7. **Fixing real failures** — testing on real documents exposed retrieval gaps, each
   fixed with a measured change: hybrid BM25 + vector search for exact names and
   numbers, page-aware PDF parsing with tables kept whole, OCR for scanned pages,
   document-level metadata on every chunk, and conversation memory with guardrails.
   See section 10.

### The agent (lead with this)

The answer engine is a **LangGraph `StateGraph`** with four nodes —
`interpret_query → retrieve → rerank → synthesize` — plus a **conditional edge
after rerank** that can loop back to `retrieve`.

- **interpret_query**: one LLM call that returns a keyword-rich search query *and* a
  standalone version of the question (so "and what tech did he use?" becomes a full
  question using the conversation history). The question a human asks and the string
  that searches best aren't the same thing.
- **retrieve**: embeds the query and runs a **hybrid** search, vector kNN fused with
  BM25 keyword search by reciprocal rank, **scoped to the user's own documents** (that's
  where tenant isolation lives) and optionally to an upload-date range. Starts at
  `top_k = 8`.
- **rerank**: Cohere's cross-encoder rescoring of the user's own question (not the
  keyword rewrite), keeping the **top 3**. Classic two-stage design: retrieval for
  recall, reranker for precision.
- **route_after_rerank**: if the best reranked chunk scores below **0.1** (a tuned,
  configurable threshold) and we haven't hit the **2-attempt cap**, loop back to
  retrieve with a **wider** `top_k` (8 → 14 → 20); otherwise synthesize. If nothing was
  retrieved at all, it skips the retry, since a wider search can't add results.
- **synthesize**: a system prompt with explicit data boundaries, then the chunks inside
  `<context>` tags and the question inside `<question>`. The model cites chunks as
  `[1]`, `[2]`, or replies `NOT_FOUND`, which becomes a plain "couldn't find it" with
  no citations.

The two judgment calls I'd emphasize: the loop is **bounded** (an unbounded agent
loop just burns API calls), and the retry **does something different** (widening
`top_k`, not re-running the same query).

### Storage

**PostgreSQL + pgvector** by default: embeddings live in-row on the `chunks` table.
One datastore, transactional consistency between a doc and its chunks, no extra infra.
Trade-off: it won't beat a dedicated vector DB at huge scale, which I'd acknowledge up
front. The vector store sits behind a small interface (`vectorstore/`), so **Chroma** can
replace it with one setting (`VECTOR_STORE=chroma`); users, documents and history always
stay in Postgres. The two stores are not migrated into each other.

---

## 3. Extended Technical Deep-Dive

### 3.1 System architecture

Backend is a FastAPI app under `backend/app/`, organized in clean layers:

| Layer | Path | Responsibility |
|---|---|---|
| API | `api/` | HTTP routers: `auth`, `documents`, `query` |
| Agents | `agents/` | The LangGraph pipeline (AI core) |
| Ingestion | `ingestion/` | parse → chunk → embed |
| Core | `core/` | config, LLM client, security, DI deps |
| DB | `db/` | SQLAlchemy models, session, init |
| Eval | `eval/` | RAGAS evaluation (two-phase) |
| Schemas | `schemas.py` | Pydantic request/response DTOs |

`main.py` builds the app, enables INFO logging, adds CORS (allowing the Vite
frontend at `http://localhost:5173`), mounts the three routers, and exposes
`/health`.

The API contract (`schemas.py`) keeps request/response shapes separate from the
ORM models on purpose — storage shape and API shape are allowed to diverge. The
`AskResponse` carries `{question, answer, citations[], trace[]}`, so the trace is a
first-class part of the response.

### 3.2 Agent architecture (full detail)

**Framework:** LangGraph ≥ 1.2. Defined in `agents/graph.py`.

**Shared state** — one `TypedDict` passed between all nodes:

```python
class AgentState(TypedDict):
    question: str
    user_id: str
    history: List[dict]                  # earlier {question, answer} turns, oldest first
    interpreted_query: Optional[str]     # keyword query for retrieval
    standalone_question: Optional[str]   # follow-up rewritten to stand alone
    retrieved_chunks: Optional[List[dict]]
    reranked_chunks: Optional[List[dict]]
    retrieval_attempts: int
    date_from: Optional[str]             # upload-date filter, inclusive ISO dates
    date_to: Optional[str]
    answer: Optional[str]
    citations: Optional[List[dict]]
    trace: List[dict]   # one entry per node: {stage, summary, detail}
```

**LangGraph idiom to mention:** nodes don't mutate state in place — each returns a
*partial dict* that LangGraph merges. That's why `trace` is accumulated with
`state.get("trace", []) + [entry]` in every node.

**Graph construction** (`build_graph()`):

```python
graph.set_entry_point("interpret_query")
graph.add_edge("interpret_query", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_conditional_edges(
    "rerank",
    route_after_rerank,
    {"retry": "retrieve", "synthesize": "synthesize"},
)
graph.add_edge("synthesize", END)
```

The graph is **compiled once per process** (`_graph = build_graph()` in
`api/query.py`) because LangGraph graphs are stateless and reusable. At startup it
logs its own ASCII diagram via `app.get_graph().draw_ascii()` (needs the
`grandalf` dependency).

**The conditional loop:**

```python
MAX_RETRIEVAL_ATTEMPTS = 2   # threshold: settings.rerank_confidence_threshold (default 0.1)

def route_after_rerank(state):
    threshold = get_settings().rerank_confidence_threshold
    reranked = state.get("reranked_chunks") or []
    attempts = state.get("retrieval_attempts", 0)
    top_score = max((c.get("relevance_score", 0) for c in reranked), default=0)
    if not state.get("retrieved_chunks"):
        return "synthesize"      # nothing retrieved: a wider search can't help
    if not reranked or (top_score < threshold and attempts < MAX_RETRIEVAL_ATTEMPTS):
        return "retry"
    return "synthesize"
```

On the retriever side, `top_k` widens per attempt:

```python
BASE_TOP_K = 8
WIDEN_PER_ATTEMPT = 6
top_k = BASE_TOP_K + (attempts * WIDEN_PER_ATTEMPT)   # 8 -> 14 -> 20
```

**Per-node summary:**

| Node | File | Input → Output | What it does |
|---|---|---|---|
| interpret_query | `query_interpreter.py` | `question`, `history` → `interpreted_query`, `standalone_question` | One LLM call: keyword query plus a standalone rewrite of follow-ups |
| retrieve | `retriever.py` | `interpreted_query` → `retrieved_chunks` | Hybrid vector + BM25 (RRF), user-scoped, optional date range, widening top_k |
| rerank | `reranker.py` | `retrieved_chunks` → `reranked_chunks` | Cohere `rerank-english-v3.0` on the standalone question, keep top 3 |
| synthesize | `synthesizer.py` | `reranked_chunks`, `history` → `answer`, `citations` | Bounded prompt; cites `[n]` or replies `NOT_FOUND` |

The retriever enforces **multi-tenant isolation** with
`.filter(Document.owner_id == user_id)` — a user can only ever retrieve from their
own documents.

### 3.3 RAG & ingestion

**Parsing** (`ingestion/parser.py`):
- **Markdown**: reads file, resolves FastAPI-style `{* path/to/file.py *}` snippet
  macros by inlining the referenced Python source *before* chunking ("resolve-
  then-chunk"), so code and its explanation stay in the same chunk. Missing files
  become visible `[code omitted: ...]` placeholders, never silent drops.
- **HTML**: BeautifulSoup, strips `script/style/nav/footer`, extracts text.
- **PDF**: pdfplumber extracts text page by page and turns each detected table into
  a Markdown table (columns preserved); page and table markers are emitted for the
  chunker. Pages with no real text layer (a scan's only text is often its page
  number) are OCR'd locally with RapidOCR and rebuilt into rows
  (`Date | Description | Branch | Venue`). pypdf plain text is the fallback, and the
  fallback is logged loudly because it loses page numbers and tables.

**Chunking** (`ingestion/chunker.py`):
- `MAX_CHUNK_CHARS = 1800`, `OVERLAP_CHARS = 200`.
- **Structure-aware for Markdown/HTML**: split on H1–H3 headers so a chunk never
  straddles unrelated sections, and the header becomes citation metadata. A guard
  (`_real_header_matches`) ignores `#` lines inside fenced code blocks so a bash
  comment isn't mistaken for a heading.
- **Page-aware for PDF**: prose chunks are labelled `p.N`, each table stays whole in
  its own chunk (`Table p.N`, split by rows with the header repeated if it is too
  long), and OCR'd pages become `OCR p.N`. Chunks with no letters (just a page
  number) are dropped.
- **Fixed-size + overlap fallback** for any header section still over 1800 chars,
  and for PDFs parsed without page markers.
- **Context prefix**: every chunk is stored as
  `[file | kind | subject | topics: ... | section]` followed by its text. The
  document-level fields come from one LLM call per upload. Without them a chunk
  from the middle of a resume never contains the person's name.

**Why those numbers:**
- `1800` chars — big enough to hold a full API-reference section (answer stays
  whole), small enough to keep retrieval precise and fit reranker/LLM context.
- `200`-char overlap — preserves continuity across boundaries so an answer
  spanning a split isn't cut in half.

**Embedding** (`ingestion/embedder.py`): provider-switchable like the LLM client.
The default provider `anthropic` (like `groq`, which has no embeddings API either) routes
to **VoyageAI `voyage-4-lite`**; OpenAI uses `text-embedding-3-small` and Gemini
`gemini-embedding-001`. All are asked for 512 dimensions to match the pgvector column.
Vectors from different models are not comparable, so a provider switch means
re-uploading documents.

**Retrieval** is **hybrid**: vector kNN plus BM25 keyword search (in-process, per user,
cached for 60 seconds), merged by reciprocal rank fusion, then reranked by a
cross-encoder. Embeddings match meaning but are weak on exact names and numbers; BM25
covers that gap. The two-stage split is deliberate: cheap wide recall, then expensive
accurate precision on a small candidate set.

### 3.4 LLM & embeddings config

`core/llm_client.py` `get_llm()` returns a LangChain chat model (drops straight
into LangGraph nodes), looked up by the `LLMProvider` enum in `settings.llm_provider`.
A typo in `LLM_PROVIDER` fails when the app starts, and both the LLM client and the
embedder check at import that every provider has an entry:

| Provider | Default model | Notes |
|---|---|---|
| anthropic (default) | `claude-sonnet-4-5` (configurable) | via `langchain-anthropic`; embeds with VoyageAI |
| openai | `gpt-4o-mini` | via `langchain-openai` |
| gemini | `gemini-3.5-flash` | via `langchain-google-genai` |
| groq | `groq/compound` | via `langchain-groq`; embeds with VoyageAI |

All at **temperature 0** for determinism. Nodes never import a provider SDK
directly — swapping providers = editing only `llm_client.py` / `embedder.py`.

Config (`core/config.py`) is loaded from the single `documind/.env` (found from the
file's own location, so it works from any directory) via `pydantic-settings`, cached
with `lru_cache`. `embedding_dim = 512` matches the pgvector `Vector(512)` column.

### 3.5 Data storage

**PostgreSQL + pgvector**, embeddings stored in-row. `DATABASE_URL` uses the
psycopg3 driver. `db/init_db.py` is idempotent: it creates the extension and tables,
then applies `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migrations, because
`create_all` never alters existing tables.

Four tables (`db/models.py`):

| Table | Key columns |
|---|---|
| `users` | id (UUID), email (unique), hashed_password, created_at |
| `documents` | id, owner_id (FK), filename, doc_type, **doc_metadata (JSONB: subject, kind, topics)**, created_at |
| `chunks` | id, document_id (FK), content, **embedding Vector(512)**, section |
| `query_logs` | id, user_id, **conversation_id**, question, answer, created_at |

With pgvector the `chunks` table *is* the vector store, searched by `cosine_distance`;
with `VECTOR_STORE=chroma` the chunks and vectors live in a local Chroma collection
instead (each vector carries `owner_id`, `filename`, `section`, subject/kind/topics and
an `uploaded_ts`). No ANN index (IVFFlat/HNSW) yet; exact scan is fine at this scale.
The upload-date filter uses `documents.created_at` (pgvector) or `uploaded_ts` (Chroma).

### 3.6 Evaluation (RAGAS, two-phase)

Metrics: **faithfulness, answer relevancy, context precision, context recall**,
against a hand-written `EVAL_SET` with ground-truth references.

Two phases in two separate virtualenvs because RAGAS pulls an old
`langchain-community` needing `langchain-core <1`, which conflicts with LangGraph
needing `langchain-core >=1.4`:

- **Phase 1 — `collect.py`** (app venv): invokes the real agent graph for every
  eval question, writes plain JSON (no LangChain objects).
- **Phase 2 — `score.py`** (RAGAS-only venv, zero `app` imports): reads the JSON,
  builds a RAGAS `EvaluationDataset`, scores it. Judge LLM is Groq
  `llama-3.3-70b-versatile`, passed explicitly so RAGAS doesn't default to an
  OpenAI judge. `answer_relevancy` also needs an embeddings model, supplied via
  VoyageAI. (Honest caveat: the scorer has not been run end to end, the eval set is
  only 4 questions, and the VoyageAI wrapper likely needs a LangChain adapter. See
  `docs/EVAL.md`.)

Splitting generation from scoring sidesteps the conflict and is cleaner anyway —
"produce answers" and "grade answers" needn't share a dependency tree.

### 3.7 Auth & security

Stateless **JWT bearer auth**. Passwords hashed with bcrypt (passlib); tokens
issued with python-jose (HS256), 24h expiry, payload `{sub: user_id, exp}`.
`core/deps.py get_current_user` is an `HTTPBearer` dependency that decodes the
token and 401s on invalid/expired/missing user. Every documents/query endpoint
depends on it, and tenant isolation is enforced by filtering on the current user
in every query.

### 3.8 Deployment

Dockerfile: `python:3.11-slim`, `pip install .` from pyproject, runs
`uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`. docker-compose adds a
`db` service (hostname `db`). Frontend is a Vite app on `:5173`.

---

## 4. One-Page Cheat Sheet

```
PROJECT     DocuMind — multi-tenant RAG Q&A over technical docs (FastAPI corpus)
STACK       Python 3.11 · FastAPI · LangGraph · Postgres+pgvector (Chroma optional) · Docker · Vite FE

AGENT (LangGraph StateGraph)
  interpret_query → retrieve → rerank → [route] → synthesize → END
                        ^__________________|   (retry, wider top_k)
  State: TypedDict, nodes return partial dicts (merged), trace accumulated per node
  Loop:  retry if top rerank score < 0.1 (configurable) AND attempts < 2; none if nothing retrieved
  top_k: 8 → 14 → 20  (BASE 8 + 6/attempt)
  Memory: conversation_id → last 4 Q&A turns; follow-ups rewritten to standalone questions

RETRIEVAL   Hybrid: vector kNN + BM25 keyword (RRF) → Cohere rerank-english-v3.0 (top 3)
            User-scoped (owner_id), optional upload-date range
            Reranker sees the user's own question, not the keyword rewrite

ANSWERING   <context>/<question> data tags, "data not instructions" system prompt,
            [n] citations, NOT_FOUND → plain "couldn't find it" (no citations), 2000-char question cap

MODELS      LLM     anthropic (default; pluggable: openai, gemini, groq; LLMProvider enum), temp=0
            Embed   VoyageAI voyage-4-lite for anthropic/groq (openai/gemini use their own), 512d
            Rerank  Cohere rerank-english-v3.0
            Judge   Groq llama-3.3-70b-versatile (eval only)

CHUNKING    1800 chars max, 200 overlap, every chunk prefixed [file | kind | subject | topics | section]
            MD/HTML → split on H1-H3 headers (code-fence-aware)
            PDF → page-aware (p.N), tables kept whole as Markdown (Table p.N), scans OCR'd (OCR p.N)
            Why: section-sized = whole answers; overlap = continuity; prefix = chunks know who/what they're about

STORAGE     Postgres + pgvector (embeddings in-row), 4 tables:
            users · documents(doc_metadata) · chunks(embedding Vector(512)) · query_logs(conversation_id)
            Chroma via VECTOR_STORE=chroma; no ANN index yet (exact scan)

EVAL        RAGAS — faithfulness, answer_relevancy, context_precision, context_recall
            Two phases / two venvs (collect in app venv, score in RAGAS venv) to dodge dep conflict

AUTH        JWT HS256 (python-jose), bcrypt (passlib), 24h expiry, HTTPBearer dep

TRANSPARENCY  Each node emits {stage, summary, detail}; trace returned in AskResponse → UI renders real pipeline
```

---

## 5. Agent Graph Diagram

![DocuMind LangGraph agent graph](images/agent-graph.svg)

ASCII fallback (same graph, for terminals / plain-text contexts):

```
                        ┌─────────────────────┐
          user question │   interpret_query   │  LLM rewrites raw question
          ─────────────▶│  (query_interpreter)│  + standalone follow-up question
                        └──────────┬──────────┘
                                   │ interpreted_query, standalone_question
                                   ▼
                        ┌─────────────────────┐
                        │      retrieve       │  hybrid vector + BM25 (RRF),
                   ┌───▶│    (retriever)      │  user-scoped, top_k = 8+6·attempts
                   │    └──────────┬──────────┘
                   │               │ retrieved_chunks
                   │               ▼
                   │    ┌─────────────────────┐
                   │    │       rerank        │  Cohere rerank-english-v3.0,
                   │    │     (reranker)      │  keep top 3 + relevance_score
                   │    └──────────┬──────────┘
                   │               │ reranked_chunks
                   │               ▼
                   │        ╔═══════════════╗
          "retry"  │        ║ route_after_  ║   top score < 0.1
      (wider top_k)└────────║    rerank     ║   AND attempts < 2  ? ─▶ retry
                            ╚═══════╤═══════╝                     else synthesize
                                    │ "synthesize"
                                    ▼
                        ┌─────────────────────┐
                        │     synthesize      │  grounded answer (temp=0),
                        │   (synthesizer)     │  ONLY from context + citations
                        └──────────┬──────────┘
                                   │ answer, citations, trace
                                   ▼
                                  END
```

---

## 6. Mock Interview Q&A

**Q: Why LangGraph instead of a simple chain or LangChain `SequentialChain`?**
Because the flow isn't linear — it has a conditional retry loop. LangGraph models
the pipeline as a state machine with explicit nodes and edges, so the retrieval
retry is a first-class edge (`add_conditional_edges`) rather than imperative
control flow buried in a function. It also makes the per-node trace natural: each
node returns a partial state update, and the merged `trace` list is the audit log.

**Q: Walk me through what happens on an ambiguous question.**
`interpret_query` cleans it up, `retrieve` pulls the top 8 chunks, `rerank`
rescores them with Cohere. If the best score is below 0.1 and we're under 2
attempts, `route_after_rerank` returns `"retry"`, which loops back to `retrieve`
with `top_k` widened to 14 (then 20). If it still can't clear the bar, it
synthesizes with what it has, and because the prompt tells the model to reply
`NOT_FOUND` when the chunks don't contain the answer, a weak context yields an
honest "I couldn't find that in your documents" rather than a hallucination. (The
threshold started at 0.5. Real runs showed Cohere scores for correct answers sitting
around 0.1 to 0.3, so 0.5 caused needless retries; Cohere scores aren't calibrated
probabilities, which is why the value is configurable and was tuned from logged scores.)

**Q: Why rerank at all if you already have vector similarity?**
Vector cosine distance measures embedding proximity, which is good for recall but
noisy for precision — semantically close isn't the same as *answers the question*.
Cohere's reranker is a cross-encoder: it looks at the query and chunk together and
scores actual relevance. So I use vectors to cheaply narrow 1000s of chunks to 8,
then spend the expensive cross-encoder only on those 8 and keep the best 3.

**Q: Why 1800 characters for chunks? How did you pick it?**
It's a balance. Too small and a single API-reference section gets fragmented
across chunks, so no single retrieved chunk contains the whole answer. Too large
and retrieval gets imprecise (lots of irrelevant text per chunk) and you waste
context budget. 1800 chars is roughly one documentation section. For structured
docs I split on headers first so chunks align with semantic units, and only fall
back to fixed-size windowing when a section is oversized. PDFs are chunked per page,
with tables kept whole so a row of numbers never loses its column headers.

**Q: Why overlap?**
A fixed-size split can cut a sentence or a code block in half, so an answer that
lives on the boundary would be split across two chunks and neither alone would
retrieve well. A 200-char overlap means boundary content appears in both
neighboring chunks, preserving continuity.

**Q: Where are embeddings stored? Why not a dedicated vector DB like Pinecone?**
In Postgres via pgvector, in-row on the `chunks` table. For this scale a single
datastore is simpler to operate, gives transactional consistency between a
document and its chunks, and avoids syncing two systems. The trade-off is that a
dedicated vector DB would scale better and offers managed ANN indexing — if the
corpus grew, I'd add an HNSW/IVFFlat index first, and only move off Postgres if
that stopped being enough.

**Q: How do you prevent hallucination?**
Several layers: temperature 0; a system prompt that says to answer only from the
chunks and to reply `NOT_FOUND` (mapped to a plain "couldn't find it" with no
citations) when they don't contain the answer; an optional rerank-score gate that
skips the LLM entirely when even the best chunk is weak; and inline `[n]` citations
tying each claim to a source doc and section so a reviewer can verify it. The
retrieval retry also reduces the "weak context → guess" failure mode. I measured the
limits too: a correct "not found" is only as good as retrieval, so a scanned table
that was never OCR'd produced a confident but wrong "not found" until OCR was added.

**Q: How is it multi-tenant? Could one user read another's docs?**
No. Auth is JWT; every protected endpoint resolves the current user via an
`HTTPBearer` dependency, and every query — upload, list, delete, retrieve,
history — filters on `owner_id == current_user`. The retriever specifically joins
chunks to documents and filters `Document.owner_id == user_id`, so vector search
can only ever return the requesting user's chunks.

**Q: How did you evaluate it, and why two virtualenvs?**
RAGAS with faithfulness, answer relevancy, context precision, and context recall
against a hand-written ground-truth set. Two venvs because RAGAS transitively
needs an older `langchain-core` that conflicts with LangGraph's. So Phase 1
generates answers through the real agent graph and dumps plain JSON in the app
venv; Phase 2 scores that JSON in a RAGAS-only venv with zero app imports. It
dodges the conflict and is a clean separation — generation and grading don't need
to share dependencies. The judge is Groq's llama-3.3-70b, passed explicitly so
RAGAS doesn't silently fall back to an OpenAI judge.

**Q: How would you scale this?**
Add an ANN index (HNSW) on the pgvector column; move ingestion off the request
path into a background worker/queue; cache query embeddings and frequent answers;
batch-embed uploads; and if the corpus got very large, consider a dedicated vector
store. I'd also persist citations/trace per query-log row so history replays are
complete.

**Q: What would you change if you did it again?**
Build the evaluation set and the real-document test cases earlier (the failures that
shaped hybrid search, OCR and the rerank threshold only showed up on real files),
run ingestion in a background queue, store citations and traces with each query log,
and put keyword search in Postgres full-text search instead of an in-process BM25
index so it scales past thousands of chunks per user.

---

## 7. Known Gaps / "What I'd Improve"

Volunteering these shows seniority:

Fixed since this guide was first written (so don't claim them as open): config drift
between code and `.env.example`, the `score.py` `type`/`category` KeyError, the
synthesizer trace naming the wrong model, and vector-only retrieval (now hybrid).

Still open:
- **Eval is thin**: 4 questions, never run end to end; the VoyageAI judge-embeddings
  wrapper in `score.py` likely needs a LangChain adapter, and its docstring still says
  Gemini. See `docs/EVAL.md`.
- **Scaling**: no ANN index on pgvector; ingestion (including OCR at 5 to 10 seconds
  per scanned page) is synchronous in the request; the BM25 index is built in memory
  per user and cached for 60 seconds.
- **History replay**: `query_logs` stores question, answer and conversation id, but not
  citations or traces, so replayed history lacks them.
- **Prompt-injection defence is mitigation, not prevention**: delimiters, a data-not-
  instructions rule and logging reduce risk; a determined attack on the model can still
  get through.
- **OCR limits**: reads text and rows, not table borders or merged cells, and can
  misread characters (`II` read as `Il`).
- **Abbreviations**: the reranker can miss a match like "AI students" vs. "AIDS";
  abbreviation expansion or sending more than 3 chunks to the model would help.
- **Document metadata is an LLM guess** from the first ~6000 characters and can be wrong;
  it is skipped when there's no real text.
- **Frontend** doesn't yet send `conversation_id` or date filters (`docs/FRONTEND.md`).
- **Embeddings are not versioned**: nothing records which model produced a stored
  vector, so a provider switch silently makes old vectors incomparable until re-upload.

---

## 8. RAGAS Metrics — Interpretation Guide

DocuMind's eval reports four RAGAS metrics, each scored **0.0–1.0** (higher is
better). Two measure the **generator** (how good the answer is given the context)
and two measure the **retriever** (how good the context is). Knowing which is
which lets you diagnose *where* a failure lives.

| Metric | What it measures | Component | What a low score means |
|---|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? (hallucination check) | Generator | The LLM is inventing facts not in the context — tighten the grounding prompt, lower temperature (already 0), or reduce context noise |
| **Answer relevancy** | Does the answer actually address the question (not padded/evasive)? | Generator | The answer drifts, over-explains, or dodges — refine the synthesis prompt |
| **Context precision** | Are the *relevant* chunks ranked at the top of what was retrieved? | Retriever | Reranker is surfacing noise — the rerank stage or `RERANK_KEEP` needs tuning |
| **Context recall** | Did retrieval actually fetch the chunks needed to answer? | Retriever | The right chunk never got retrieved — widen `top_k`, improve chunking, or fix embeddings |

### How to read the two axes together

```
                 CONTEXT (retriever)
                 low recall/precision        high recall/precision
ANSWER      ┌──────────────────────────┬──────────────────────────┐
(generator) │  Retrieval is the        │  System is healthy.       │
 high faith │  bottleneck. The LLM is  │  Good context, grounded,  │
 + relevancy│  honest about weak       │  on-topic answers.        │
            │  context (good!) but     │                           │
            │  can't answer well.      │                           │
            ├──────────────────────────┼──────────────────────────┤
 low faith  │  Worst case — bad        │  Generation is the        │
 or         │  context AND the LLM     │  bottleneck. Good chunks  │
 relevancy  │  hallucinates/drifts     │  retrieved but the answer │
            │  on top of it.           │  ignores them or rambles. │
            └──────────────────────────┴──────────────────────────┘
```

### Rough score bands (rule-of-thumb, not official thresholds)

| Band | Score | Reading |
|---|---|---|
| Strong | **≥ 0.85** | Production-ready on that dimension |
| Acceptable | **0.70 – 0.85** | Usable; worth tuning |
| Weak | **0.50 – 0.70** | Noticeable quality problems; investigate |
| Failing | **< 0.50** | That component is broken for the test set |

> These bands are heuristics for interview discussion. Absolute numbers vary with
> the judge model, dataset difficulty, and corpus — what matters is **trend over
> time** and **relative gaps between the four metrics**, not hitting a magic number.

### Which DocuMind design choice each metric validates

- **Faithfulness** → validates the strict grounding prompt + temperature 0 in the
  `synthesize` node.
- **Answer relevancy** → validates the `interpret_query` rewrite (a cleaner query
  tends to produce more on-point answers).
- **Context precision** → validates the **Cohere rerank** stage (its whole job is
  pushing relevant chunks to the top).
- **Context recall** → validates the **chunking strategy** and the **retrieval
  retry loop** (widening `top_k` on low confidence is a direct recall lever).

### How I'd present results in an interview

> "I don't quote a single accuracy number because RAGAS gives four, and the useful
> signal is the *gap* between them. If faithfulness is high but context recall is
> low, my retriever is the bottleneck — the LLM is being honest about weak context
> rather than hallucinating, which is exactly what the grounding prompt is for. If
> recall is high but faithfulness is low, the context is fine and the generator is
> the problem. That diagnosis tells me which part of the pipeline to tune next,
> and the retrieval retry loop plus the rerank stage are the two levers I built
> specifically to move the retriever-side metrics."

---

## 8. RAGAS Metrics — How to Interpret Scores

All four RAGAS metrics are normalized to **0.0–1.0 (higher is better)**. There is
no universal pass/fail line — the thresholds below are practical bands used in RAG
projects, not official constants. The key skill in an interview is explaining
**what each metric isolates** and **what a low score points you to fix**.

### 8.1 The two axes

RAGAS metrics split cleanly into two groups, and knowing which is which is the
whole trick:

| Axis | Metrics | What it grades | If low, the problem is in... |
|---|---|---|---|
| **Retrieval quality** | context_precision, context_recall | Did we fetch the *right* chunks? | chunking, embeddings, top_k, reranker |
| **Generation quality** | faithfulness, answer_relevancy | Did the LLM *use* them well? | the synthesis prompt, the model, grounding |

This mapping is the single most useful thing to say: a low **generation** score
with high **retrieval** scores means the context was there and the model fumbled
it; a low **retrieval** score means the model never had a chance.

### 8.2 Metric-by-metric

**Faithfulness** — *Is every claim in the answer supported by the retrieved
context?* Measures hallucination. RAGAS breaks the answer into atomic claims and
checks each against the context.
- Interpretation bands: `≥0.90` strong · `0.75–0.90` acceptable · `<0.75`
  hallucination risk.
- A low score here is the scariest for a docs assistant — it means the model is
  inventing facts. **Fix:** tighten the grounding prompt, lower temperature (already
  0 here), or the context is too thin so the model fills gaps (then look at recall).

**Answer Relevancy** — *Does the answer actually address the question?* RAGAS
reverse-generates questions from the answer and measures their similarity to the
original question (this is the metric that needs an embeddings model).
- Bands: `≥0.85` strong · `0.70–0.85` acceptable · `<0.70` off-topic / padded.
- Low score = rambling, partial, or evasive answers. **Fix:** prompt the model to
  be direct and complete; check the query-interpretation node isn't distorting
  intent.

**Context Precision** — *Of the chunks we retrieved, how many are actually
relevant, and are the relevant ones ranked near the top?* It's a signal-to-noise
and ranking metric.
- Bands: `≥0.80` strong · `0.60–0.80` acceptable · `<0.60` noisy retrieval.
- Low score = too much junk in the context window. **This is exactly what the
  Cohere reranker exists to raise** — it reorders so relevant chunks sit on top and
  trims to the top 3. **Fix:** rerank harder (smaller `RERANK_KEEP`), smaller
  chunks, better embeddings.

**Context Recall** — *Did retrieval fetch everything needed to answer?* Compares
retrieved context against the ground-truth reference (this is why the eval set
needs hand-written references).
- Bands: `≥0.80` strong · `0.60–0.80` acceptable · `<0.60` missing information.
- Low score = the answer-bearing chunk never got retrieved. **This is what the
  conditional retrieval retry loop targets** — widening `top_k` (8→14→20) on a weak
  pass raises recall. **Fix:** raise `top_k`, improve chunking so answers aren't
  fragmented, check hybrid (BM25 + vector) is on, and check the document was actually
  extracted (a scan without OCR has nothing to retrieve).

### 8.3 The precision/recall tension (good thing to volunteer)

Precision and recall pull against each other. Raising `top_k` fetches more chunks
→ recall up, but drags in noise → precision down. DocuMind's design resolves this
deliberately: **retrieve wide for recall** (vector kNN, widening on retry), then
**rerank narrow for precision** (Cohere cross-encoder trims to the top 3). So the
retry loop pushes recall and the reranker protects precision — the two stages
tune opposite metrics.

### 8.4 How to read a result table in the interview

If asked "your eval came back like this, what do you do?":

| faithfulness | ans_relevancy | ctx_precision | ctx_recall | Diagnosis & action |
|---|---|---|---|---|
| high | high | high | high | Shipping-ready; sample a few answers manually to confirm. |
| **low** | high | high | high | Model hallucinates despite good context → tighten grounding prompt / try a stronger model. |
| high | **low** | high | high | Answers wander off-question → fix synthesis prompt; check query interpreter. |
| high | high | **low** | high | Context is noisy → rerank harder, shrink chunks, trim `RERANK_KEEP`. |
| high | high | high | **low** | Right answer never retrieved → raise `top_k`, fix chunking, confirm hybrid retrieval is on and the source was extracted (OCR). |
| low | low | **low** | **low** | Retrieval is the root cause → start at ingestion/embeddings before touching the LLM. |

The habit this shows: **diagnose retrieval before generation.** If the context is
wrong, no prompt tuning saves you — so read `context_recall` and
`context_precision` first, and only blame the LLM (`faithfulness`,
`answer_relevancy`) once you've confirmed the right chunks were in the window.

### 8.5 Caveats worth naming

- Scores are **LLM-judged** (Groq llama-3.3-70b here), so they're noisy on tiny
  eval sets — report them as directional, not exact, and keep the eval set fixed
  for comparability (that's the point of the hand-written `EVAL_SET`).
- `answer_relevancy` depends on an embeddings model, so its absolute value shifts
  if you change embedding providers — compare runs that used the same judge setup.
- A perfect score on a 4–5 question set means little; the harness is a regression
  guard (did a change make things worse?) more than an absolute grade.

---

## 9. Live Demo Script

A tight, narratable sequence for screen-sharing with an interviewer. The whole
run is ~4 minutes. The goal isn't to show the UI — it's to make the **agent
pipeline visible**, so the money moment is reading the `trace` out loud,
especially if a retry fires.

### 9.1 Before the call (setup, don't do this live)

```bash
cp .env.example .env
# fill in: JWT_SECRET_KEY, ANTHROPIC_API_KEY, VOYAGEAI_API_KEY, COHERE_API_KEY
docker compose up --build -d
docker compose exec backend python -m app.db.init_db   # extension, tables, column migrations (idempotent; re-run after every pull)
```

Pre-flight checklist so nothing fails on camera:
- `GET http://localhost:8000/health` returns `{"status":"ok"}`
- Swagger UI loads at `http://localhost:8000/docs`
- Frontend loads at `http://localhost:5173`
- Have a sample doc ready from `data/sample_docs/fastapi/` (e.g.
  `tutorial_cors.md` or `tutorial_handling-errors.md`)
- Have **one easy question** and **one deliberately vague question** written down
  (the vague one is to try to trigger the retry loop — see 9.4)

> Tip: you can demo entirely through **Swagger UI** (`/docs`) if the frontend is
> flaky. It shows the raw JSON response including the full `trace`, which is
> actually better for narrating the pipeline.

### 9.2 The narration arc (what to say while clicking)

Open with one sentence of framing before you touch anything:

> "I'll register a user, upload a doc so it gets ingested into pgvector, then ask
> a question — and the thing to watch is the trace in the response, because that's
> the actual LangGraph pipeline, not a mock."

### 9.3 Step-by-step

**Step 1 — Register (`POST /auth/register`)**
Body: `{"email": "demo@example.com", "password": "demopassword"}`
> "Multi-tenant from the start — this returns a user id, and everything from here
> is scoped to this user."

**Step 2 — Login (`POST /auth/login`)**
Same credentials → returns `{"access_token": "...", "token_type": "bearer"}`.
> "JWT bearer token. In Swagger I'll paste this into Authorize so every call is
> authenticated." (Click **Authorize**, paste `Bearer <token>` / the token.)

**Step 3 — Upload a document (`POST /documents/upload`)**
Attach `tutorial_cors.md`. Returns the document record.
> "This upload runs the whole ingestion pipeline synchronously — parse, chunk on
> headers at 1800 chars, embed each chunk with VoyageAI, and store the vectors
> in-row in pgvector. One request, and the doc is queryable."
Optionally hit `GET /documents/` to show it listed.

**Step 4 — Ask a question (`POST /query/ask`) — the main event**
Body: `{"question": "How do I enable CORS in FastAPI?"}`
Point at the response and read the three parts:
> "Three things come back. The **answer**, grounded in the doc. The **citations**
> — doc plus section, so it's verifiable. And the **trace** — this is the real
> pipeline: `interpret` rewrote my question into a search query, `retrieve` pulled
> N candidate chunks from pgvector, `rerank` scored them with Cohere and kept the
> top 3, and `synthesize` wrote the answer. Each entry is emitted by its node."

**Step 5 — Show grounding honesty (optional, strong)**
Ask something the doc can't answer:
`{"question": "What is FastAPI's pricing model?"}`
> "Because the synthesis prompt is strictly grounded, it says the docs don't cover
> this instead of hallucinating a number. That's the behavior you want in a docs
> assistant."

**Step 6 — History (`GET /query/history`)**
> "Queries are logged per user — this is the history endpoint. Note it currently
> stores just question and answer, so citations and trace are empty on replay —
> that's one of my listed next steps."

### 9.4 Trying to trigger the retry loop (the standout moment)

The conditional loop fires when the top Cohere rerank score is **below the
threshold (default 0.1)** and attempts are **< 2**. Scores for good answers often sit
around 0.1 to 0.3, so a question has to be quite weak to trigger it. To make it
likely (and to see the scores), keep `RERANK_DEBUG=true`:
- Ask a **vague or tangential** question against a **small** uploaded corpus — e.g.
  upload only `tutorial_cors.md`, then ask something only loosely related like
  `"how does the framework handle browser preflight security stuff?"`.
- In the returned `trace`, look for a **second `retrieve` entry** whose detail
  shows `top_k` widened (`retry #1, widened top_k`, `top_k=14`).

> "Watch — the first rerank came back weak, under the 0.1 confidence threshold, so
> the router sent it back to retrieve with a wider top_k, 8 up to 14, before
> answering. That retry is bounded to two attempts so it can't loop forever. This
> is why I call it an agent and not a chain — it makes a routing decision based on
> its own intermediate confidence."

If it doesn't retry on the day, don't fish for it live — just say the above and
point to `route_after_rerank` in `agents/graph.py`. A reliable fallback is to show
the code: the conditional edge and the threshold constants tell the story on their
own.

### 9.5 If something breaks (recovery lines)

- **Upload/embed errors** (API key/quota): "That's a provider quota hiccup — the
  pipeline itself is provider-abstracted, so I'd just switch `LLM_PROVIDER` in
  config." Then fall back to walking the code.
- **Empty/weak answer**: "Small corpus, so recall is limited — this is exactly
  what context_recall measures and what the retry loop is designed to help."
- **Frontend down**: pivot to Swagger `/docs`, which is better for showing the raw
  trace anyway.

### 9.6 The 20-second code tour (have these open in tabs)

If the interviewer wants code after the demo, open these four in order:

1. `agents/graph.py` — the StateGraph, nodes, and the `route_after_rerank`
   conditional edge (the architecture in one screen).
2. `agents/retriever.py` — the user-scoped pgvector query and the
   `top_k = 8 + attempts*6` widening.
3. `ingestion/chunker.py` — header-aware chunking with the 1800/200 constants.
4. `eval/score.py` — the four RAGAS metrics and the explicit Groq judge.

> "Those four files are basically the whole story — the graph, the retrieval, the
> chunking, and how I measured it."

---

## 10. Recent Upgrades — Talking Points and Q&A

These came from testing on real documents (a resume, a scanned college circular, a
salary-style table), not from a feature list. Telling each as *symptom → cause → fix →
what I verified* is a stronger interview story than the feature itself.

### 10.1 The debugging stories

**1. Rerank scores of 0.0 on a correct answer.**
- *Symptom:* "What projects did Dharma work on?" returned a good answer but every
  rerank score was `0.0`, which also triggered a pointless retry.
- *Causes:* project chunks never contained the person's name (a resume's bullets don't
  repeat it); the reranker was given a keyword rewrite instead of the real question; and
  the retry threshold of 0.5 assumed Cohere scores behave like probabilities.
- *Fixes:* rerank on the user's own question; prefix every chunk with filename plus
  document-level metadata (subject, kind, topics, from one LLM call per upload); lower
  the threshold to 0.1 and make it configurable; add a temporary `RERANK_DEBUG` that
  prints the whole ranking.
- *Evidence:* a later resume question showed scores of 0.56 / 0.34 / 0.02 instead of
  all zeros (a different question, so directional, not a controlled comparison).

**2. A wrong "not found" on a table that was in the document.**
- *Symptom:* "Venue of project review 2 for AI students?" returned the not-found
  message; the answer was in a table in a PDF.
- *Diagnosis:* a SQL query for the venue text returned no rows, and every chunk of that
  PDF held only its page number. The "table" was an image, and the only real text layer
  on each page was the page number, which counted as content, so the upload had passed.
- *Fixes:* local OCR (RapidOCR) for pages with no usable text layer, rebuilt into rows
  (`Date | Description | Branch | Venue`); chunks that are only a page number are dropped
  (an unreadable file now gets a 400); metadata extraction is skipped when there's no
  real text, because it had been inventing a "project circular" label from the filename.
- *Verified:* on a generated image-only PDF built like the circular, not on the real
  file. OCR took about 9 seconds per page and read `II*` as `Il*`.
- *Lesson:* the guardrail did its job (it declined rather than guess); the bug was
  upstream in ingestion. Check what was *stored* before tuning prompts.

**3. A silent fallback hid missing page numbers.** Citations read `unknown` even after
the page-aware chunker shipped, because the parser quietly fell back to plain text when
`pdfplumber` wasn't installed. It now logs a warning naming the cause and recovers per
page instead of per file.

### 10.2 What was added and why

| Upgrade | Why | Trade-off |
|---|---|---|
| **Hybrid retrieval** (BM25 + vector, RRF) | Embeddings are weak on exact names and numbers | In-memory BM25 per user (cached 60 s); move to Postgres full-text search at scale |
| **Page-aware PDF + tables as Markdown** | Salary/percentage tables lose meaning when columns are flattened | Merged cells and borderless tables are only partly recovered |
| **OCR for scans** | A scanned PDF has nothing to retrieve otherwise | 5 to 10 s per page inside the upload request |
| **Document metadata prefix** | Chunks with pronouns or no subject can't be matched or understood | One extra LLM call per upload; metadata is a guess from the first ~6000 characters |
| **Conversation memory** | Follow-ups ("and what tech did he use?") need the previous turn | History is capped (4 turns), stored server-side and scoped by user |
| **Boundaries and `NOT_FOUND`** | Documents are untrusted input; the model must be allowed to say no | Mitigates prompt injection, doesn't eliminate it; can over-decline |
| **Upload-date filter** | Narrow search to documents from a period | Filters on upload date, not dates written inside documents |
| **`LLMProvider` enum** | A typo in `LLM_PROVIDER` should fail at startup, not on the first question | Doesn't validate API keys; a wrong key still fails on first call |
| **Console logging per node** | Nothing was visible under uvicorn except crashes | Question text appears in logs; use `LOG_LEVEL=WARNING` to quiet it |

### 10.3 Likely follow-up questions

**Q: How does it handle follow-up questions?**
The client sends back the `conversation_id` from the previous response. The server loads
the last four question/answer pairs for that conversation and that user, shows them to
the model as real user and assistant messages, and the interpret step rewrites the new
question into a standalone one ("What technologies did Dharma use for LoopDesk?"). That
standalone version drives keyword search and reranking, which can't see history. History
is server-side so a caller can't forge earlier "assistant" turns, and the prompt says
history only resolves references and is never a source of facts.

**Q: How do you defend against prompt injection from uploaded documents?**
Layers, none perfect. Retrieved chunks, the question and history are placed inside
`<context>`, `<question>` and conversation tags, and the system prompt says text inside
them is data, never instructions. Text that tries to close or forge those tags is
neutralised, questions are capped at 2000 characters, and phrases like "ignore previous
instructions" in a question or a chunk are logged (without echoing them). The model
still decides what to do with the text, so this reduces the risk and doesn't remove it.
Tenant isolation doesn't depend on the model at all: every search is filtered by owner
in code.

**Q: Why hybrid search instead of just better embeddings?**
Because the failure is structural. An embedding of "Priya's hike 8.5%" and "Rahul's hike
7.5%" are nearly identical, so exact tokens get lost. BM25 matches the literal name or
number. Reciprocal rank fusion merges the two rankings without needing to calibrate their
different score scales, and the reranker then sorts the merged candidates.

**Q: How do you handle scanned documents and tables?**
Per page: if the text layer has almost no letters (page numbers don't count) or the page
is mostly an image, run local OCR and rebuild reading-order rows; otherwise use the text
layer and extract tables as Markdown so columns survive. Tables stay in one chunk, or
split by rows with the header repeated. OCR is local, so no page image leaves the machine.
Limits: no merged-cell recovery, occasional character misreads, and it's slow.

**Q: Why did you change the rerank threshold from 0.5 to 0.1?**
I assumed Cohere scores were probabilities. Logged scores showed correct answers topping
out around 0.1 to 0.3, so 0.5 retried needlessly. The value is now a setting, and I used
the debug output to choose it. A principled next step is to calibrate it against the eval
set instead of by eye.

**Q: What happens when two documents contradict each other?**
The prompt tells the model not to silently pick one: say the sources disagree, give each
value with its chunk id, and mention any date or version that could explain it. That relies
on the model following the instruction and I haven't measured how often it does. A stronger
design is a separate verification step that checks claims against each chunk, at the cost
of another LLM call.

**Q: How did you debug a wrong answer?**
Follow the pipeline in order and look at what each stage actually received: the trace
(rewritten query, candidate count, scores), the rerank debug output (which chunks reached
Cohere and how they scored), and finally a SQL query against the stored chunks. That last
step found the scanned-table bug: the text simply wasn't in the database.
