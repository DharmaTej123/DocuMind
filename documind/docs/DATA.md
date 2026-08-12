# Sample Data — Step 3

## Source
Pulled from the official [FastAPI](https://github.com/fastapi/fastapi) repository
(MIT licensed) — a real, actively maintained technical documentation set. Chosen
over synthetic docs because it gives a credible eval story: real API-reference
prose, real cross-references, real formatting inconsistencies.

- Repo: `github.com/fastapi/fastapi`
- Docs pulled from: `docs/en/docs/`
- Code snippets pulled from: `docs_src/`
- Pulled: 2026-07 (sparse checkout, not a full clone)

## What's included
- **`fastapi/`** — 24 curated Markdown docs covering core API-reference topics:
  path/query/body params, dependencies, security (OAuth2/JWT), middleware, CORS,
  error handling, background tasks, testing, bigger-applications structure, and a
  few advanced topics (custom responses, additional status codes, OAuth2 scopes).
- **`fastapi_code_src/`** — the Python code files those docs reference (212 files),
  needed to resolve the snippet macros below.

## Known data quirk (feeds Step 4 chunker/parser decisions)
23 of the 24 docs embed code examples via a custom macro, not inline fenced code
blocks:
```
{* ../../docs_src/first_steps/tutorial001_py310.py *}
```
A naive Markdown chunker will treat this as a dead reference and lose all code
context — which matters a lot for API-reference docs, since the code *is* often
the answer. Two options for the parser (Step 3→4 handoff):
1. **Resolve macros at ingestion time** — replace `{* path *}` with the actual
   file content from `fastapi_code_src/` before chunking, so code ends up in the
   same chunk as its explanation.
2. **Chunk separately, link by section** — keep code chunks and prose chunks
   separate but tag them with the same `section` so retrieval can pull both.

Leaning toward option 1 (resolve-then-chunk) — simpler retrieval, and it's the
more realistic real-world scenario (most wikis/READMEs have *some* broken or
templated references you have to normalize on ingest).

## Suggested eval question seeds (for Step 7)
A few representative question types this dataset supports, spanning single-doc
lookup and cross-doc reasoning:
- Direct lookup: "How do I declare a path parameter with a type in FastAPI?"
- Cross-doc: "What's the difference between a query parameter and a path
  parameter?" (spans two docs)
- Code-dependent: "Show the code for adding a background task after returning a
  response." (only answerable if macro-resolution works)
- Multi-step: "How do I set up OAuth2 with JWT and restrict a route to a
  specific scope?" (spans oauth2-jwt.md and oauth2-scopes.md)
