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

## Local dev
```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```
