"""
Document parsing — FR3. Extracts text from PDF/Markdown/HTML input.

Implements the "resolve-then-chunk" decision from docs/DATA.md: Markdown
docs that embed code via `{* path/to/file.py *}` snippet macros get those
macros replaced with the actual file content before chunking, so code and
its explanation end up in the same chunk instead of being lost.
"""
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader

SNIPPET_MACRO = re.compile(r"\{\*\s*(\.\./)*(?P<path>[^\s*]+\.py)(\s*hl\[[^\]]*\])?\s*\*\}")


def parse_document(file_path: str, doc_type: str, code_src_root: Optional[str] = None) -> str:
    """
    Returns the document's extracted plain text, ready for chunking.

    code_src_root: base directory to resolve relative code-snippet macro
    paths against (only used for doc_type == 'markdown'). If a referenced
    file can't be found, the macro is left as a visible placeholder rather
    than silently dropped, so ingestion failures are debuggable.
    """
    if doc_type == "markdown":
        return _parse_markdown(file_path, code_src_root)
    if doc_type == "html":
        return _parse_html(file_path)
    if doc_type == "pdf":
        return _parse_pdf(file_path)
    raise ValueError(f"Unsupported doc_type: {doc_type}")


def _parse_markdown(file_path: str, code_src_root: Optional[str]) -> str:
    text = Path(file_path).read_text(encoding="utf-8")

    def resolve(match: "re.Match") -> str:
        rel_path = match.group("path")
        if code_src_root is None:
            return f"[code omitted: {rel_path}]"
        # Sample data copies docs_src/<subdir>/... into code_src_root directly
        # (see docs/DATA.md), so strip the docs_src/ prefix before joining.
        stripped = rel_path.removeprefix("docs_src/")
        candidate = (Path(code_src_root) / stripped).resolve()
        if candidate.is_file():
            code = candidate.read_text(encoding="utf-8")
            return f"```python\n{code}\n```"
        return f"[code omitted: {rel_path} not found]"

    return SNIPPET_MACRO.sub(resolve, text)


def _parse_html(file_path: str) -> str:
    html = Path(file_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


# Markers emitted for PDFs so the chunker can recover page numbers and keep
# tables intact. See ingestion/chunker.py (_chunk_pdf).
PAGE_MARK = "<<<PAGE {n}>>>"
TABLE_START = "<<<TABLE>>>"
TABLE_END = "<<<END_TABLE>>>"


def _cell(value) -> str:
    """One table cell as single-line, pipe-safe text."""
    if value is None:
        return ""
    return " ".join(str(value).split()).replace("|", "/")


def _table_to_markdown(rows: list[list]) -> str:
    """Markdown table from pdfplumber rows; first row is treated as the header."""
    cleaned = [[_cell(c) for c in row] for row in rows]
    cleaned = [r for r in cleaned if any(r)]  # drop fully empty rows
    if len(cleaned) < 2 or max(len(r) for r in cleaned) < 2:
        return ""  # a 1-row or 1-column "table" is almost always a false positive
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    lines = ["| " + " | ".join(cleaned[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    return "\n".join(lines)


def _parse_pdf(file_path: str) -> str:
    """
    Text PDFs with tables. pdfplumber finds tables and we render each as a
    Markdown table (columns preserved); the remaining page text is extracted
    with table areas excluded so numbers aren't duplicated. Output carries
    page and table markers for the chunker. Scanned PDFs (no text layer) still
    yield nothing; there is no OCR.

    Falls back to pypdf plain text if pdfplumber is unavailable or fails.
    """
    try:
        import pdfplumber
    except ImportError:
        return _parse_pdf_plain(file_path)

    try:
        parts: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                tables = page.find_tables()
                text_page = page
                for t in tables:
                    text_page = text_page.outside_bbox(t.bbox)
                text = (text_page.extract_text() or "").strip()

                block = [PAGE_MARK.format(n=number)]
                if text:
                    block.append(text)
                for t in tables:
                    md = _table_to_markdown(t.extract())
                    if md:
                        block += [TABLE_START, md, TABLE_END]
                parts.append("\n".join(block))
        return "\n\n".join(parts)
    except Exception:
        return _parse_pdf_plain(file_path)


def _parse_pdf_plain(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
