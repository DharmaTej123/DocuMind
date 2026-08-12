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


def chunk_text(text: str, doc_type: str) -> list[Chunk]:
    if doc_type in ("markdown", "html"):
        return _chunk_by_headers(text)
    return _chunk_fixed_size(text)


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
