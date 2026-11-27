# Frontend — Step 6

## Design direction
DocuMind is a developer tool for querying technical docs, so the visual
language is drawn from the tools engineers already trust — code review UIs,
doc sites, CI pipeline traces — rather than a generic SaaS/chat aesthetic.

**Tokens**
- Color: `ink #14161F` (sidebar/nav), `paper #EEF1F6` (canvas), `surface #FFFFFF`
  (cards), `signal #0F8B8D` (teal — primary interactive/retrieve), `attention
  #E7A33E` (amber — rerank/flagged), `muted #6B7280`, `danger #C1443A`.
- Type: **Space Grotesk** (display — nav, headings), **Inter** (body copy),
  **JetBrains Mono** (citations, trace labels, metadata) — the mono face is
  used deliberately wherever the UI shows something the *system* produced
  (a doc filename, a score, a stage name), separating "what the pipeline did"
  from "what it's telling you."

**Signature element — Agent Trace**
Every answer renders the actual LangGraph node sequence that produced it
(`interpret_query → retrieve → rerank → synthesize`) as an expandable strip
beneath the answer, not a generic "thinking..." spinner. Clicking a stage
shows what happened there — the rewritten query, chunk counts, rerank scores.
This is the multi-agent architecture made visible, which is also the
strongest interview talking point: most RAG demos show only the answer,
DocuMind shows *why* the answer is trustworthy.

## Pages built
- `LoginPage` / `RegisterPage` — auth (FR1)
- `AskPage` (`/`) — query composer + answer stream with citations + trace (FR4–FR9)
- `DocumentsPage` (`/documents`) — drag-and-drop upload + doc list + delete (FR2, FR10)
- `HistoryPage` (`/history`) — past queries (FR9)

## Stack
React 18 + TypeScript + Vite + Tailwind + `class-variance-authority`
(hand-built shadcn/ui-style primitives: `Button`, `Input`, `Card`, `Badge` —
not the shadcn CLI, but the same composition pattern) + `react-router-dom`.

## Current state
Builds and type-checks clean (`npm run build`). Pages currently fall back to
demo/seed data when the backend call fails, since `/auth`, `/documents`, and
`/query` are still stubs (Step 4/5 fills these in) — the fallback is clearly
commented in `AskPage.tsx` and `DocumentsPage.tsx` and should be removed once
the real endpoints are live.

## Backend changes the frontend must adopt
The backend gained features the UI does not use yet. None are implemented in `frontend/`.
- **Conversations:** `POST /query/ask` accepts an optional `conversation_id` and returns
  one. Keep it in state and send it with each question in the same chat; drop it for
  "new chat". Without it every question is a fresh conversation, so follow-ups like "and
  what tech did he use?" won't resolve. `/query/history` rows now carry `conversation_id`
  too, so history can be grouped into chats.
- **Date filter:** optional `uploaded_after` / `uploaded_before` (`YYYY-MM-DD`, inclusive).
  Invalid or reversed dates return 422.
- **"Not found" answers:** when the documents don't contain the answer the response is a
  plain message with `citations: []`. Render that as a normal answer, not an error.
- **Citation sections** can now be `p.N`, `Table p.N` or `OCR p.N` for PDFs, and
  `/documents/` returns `doc_metadata` (`subject`, `kind`, `topics`) per document.
- **Trace stages:** the retrieve step's summary now reports hybrid counts, and the
  interpret step shows the resolved follow-up question.
- **Upload latency:** scanned PDFs are OCR'd inside the upload request (about 5 to 10
  seconds per page), so the upload call needs a long timeout and a progress indicator.
- **Limits:** questions are capped at 2000 characters.

## Local dev
```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```
