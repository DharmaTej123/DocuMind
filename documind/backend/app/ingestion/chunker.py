"""
Chunking strategy — FR3. Structure-aware for Markdown/HTML (split on
headers, so a chunk never straddles two unrelated sections), fixed-size
with overlap as the fallback for PDF (which has no reliable structure once
extracted to plain text).
"""
import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 1800
OVERLAP_CHARS = 200

HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class Chunk:
    content: str
    section: str


GENERIC_SECTIONS = {"unknown", "untitled", ""}


def add_context(chunks: list[Chunk], filename: str, meta: dict | None = None) -> list[Chunk]:
    """
    Prefixes each chunk with its source and what the whole document is about, e.g.
    '[Dharma_Tej_Resume.pdf | resume | Dharma Tej | topics: LangGraph, RAG | p.2]'.
    A chunk from the middle of a document usually doesn't repeat the document's
    subject (a resume's project bullets never say the person's name), so
    embeddings, BM25 and the reranker would otherwise have nothing to match a
    name-based query against. Done once at ingestion so every stage sees it.
    meta comes from ingestion/metadata.py and may be empty.
    """
    meta = meta or {}
    for chunk in chunks:
        parts = [filename]
        if meta.get("kind"):
            parts.append(meta["kind"])
        if meta.get("subject"):
            parts.append(meta["subject"])
        if meta.get("topics"):
            parts.append("topics: " + ", ".join(meta["topics"]))
        if chunk.section and chunk.section not in GENERIC_SECTIONS:
            parts.append(chunk.section)
        chunk.content = f"[{' | '.join(parts)}]\n{chunk.content}"
    return chunks


PAGE_RE = re.compile(r"<<<PAGE (\d+)>>>")
TABLE_RE = re.compile(r"<<<TABLE>>>\n(.*?)\n<<<END_TABLE>>>", re.DOTALL)


def chunk_text(text: str, doc_type: str) -> list[Chunk]:
    if doc_type in ("markdown", "html"):
        return _chunk_by_headers(text)
    if doc_type == "pdf" and PAGE_RE.search(text):
        return _chunk_pdf(text)
    return _chunk_fixed_size(text)


def _chunk_pdf(text: str) -> list[Chunk]:
    """
    Page-aware PDF chunking. Prose is chunked fixed-size within a page and
    labelled 'p.N' for citations. Each table stays whole in its own chunk
    (labelled 'Table p.N'); tables over MAX_CHUNK_CHARS are split by rows with
    the header row repeated, so no chunk is ever a bare run of numbers.
    """
    pieces = PAGE_RE.split(text)  # ['', '1', body1, '2', body2, ...]
    chunks: list[Chunk] = []
    for i in range(1, len(pieces) - 1, 2):
        page, body = pieces[i], pieces[i + 1]
        cursor = 0
        for m in TABLE_RE.finditer(body):
            chunks += _chunk_fixed_size(body[cursor:m.start()], section=f"p.{page}")
            chunks += _chunk_table(m.group(1).strip(), page)
            cursor = m.end()
        chunks += _chunk_fixed_size(body[cursor:], section=f"p.{page}")
    return chunks


def _chunk_table(markdown: str, page: str) -> list[Chunk]:
    section = f"Table p.{page}"
    label = f"[Table from page {page}]\n"
    if len(label) + len(markdown) <= MAX_CHUNK_CHARS:
        return [Chunk(content=label + markdown, section=section)]

    lines = markdown.split("\n")
    header = "\n".join(lines[:2])  # header row + separator row
    chunks: list[Chunk] = []
    current: list[str] = []
    size = len(label) + len(header)
    for row in lines[2:]:
        if current and size + len(row) + 1 > MAX_CHUNK_CHARS:
            chunks.append(Chunk(content=label + header + "\n" + "\n".join(current), section=section))
            current, size = [], len(label) + len(header)
        current.append(row)
        size += len(row) + 1
    if current:
        chunks.append(Chunk(content=label + header + "\n" + "\n".join(current), section=section))
    return chunks


def _real_header_matches(text: str) -> list:
    """
    Finds header lines, excluding any '#'-prefixed line that falls inside a
    fenced code block (e.g. a bash comment like '# run this'), which the
    naive regex would otherwise mistake for a Markdown header.
    """
    code_spans = [(m.start(), m.end()) for m in FENCE_RE.finditer(text)]

    def in_code(pos: int) -> bool:
        return any(start <= pos < end for start, end in code_spans)

    return [m for m in HEADER_RE.finditer(text) if not in_code(m.start())]


def _chunk_by_headers(text: str) -> list[Chunk]:
    """
    Splits on H1-H3 headers first, then further splits any section that's
    still too large (e.g. a long code-heavy section) with the fixed-size
    fallback, so no single chunk blows past MAX_CHUNK_CHARS.
    """
    matches = _real_header_matches(text)
    if not matches:
        return _chunk_fixed_size(text, section="untitled")

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        section_title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(content=section_text, section=section_title))
        else:
            for sub in _chunk_fixed_size(section_text, section=section_title):
                chunks.append(sub)

    return chunks


def _chunk_fixed_size(text: str, section: str = "unknown") -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + MAX_CHUNK_CHARS, n)
        chunks.append(Chunk(content=text[start:end].strip(), section=section))
        if end == n:
            break
        start = end - OVERLAP_CHARS
    return [c for c in chunks if c.content]
