"""
Document parsing — FR3. Extracts text from PDF/Markdown/HTML input.

Implements the "resolve-then-chunk" decision from docs/DATA.md: Markdown
docs that embed code via `{* path/to/file.py *}` snippet macros get those
macros replaced with the actual file content before chunking, so code and
its explanation end up in the same chunk instead of being lost.
"""
import logging
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.core.config import get_settings
from app.ingestion import ocr

logger = logging.getLogger("documind.parser")

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


OCR_START = "<<<OCR>>>"
OCR_END = "<<<END_OCR>>>"
OCR_RICH_TEXT_CHARS = 400  # a page with this many letters of real text is trusted as-is
OCR_IMAGE_COVERAGE = 0.25  # an image covering this share of the page counts as 'mostly image'


def _image_coverage(page) -> float:
    """Share of the page area covered by embedded images (0.0 to ~1.0)."""
    area = page.width * page.height
    if not area:
        return 0.0
    covered = 0.0
    for im in page.images:
        w = min(im["x1"], page.width) - max(im["x0"], 0)
        h = min(im["bottom"], page.height) - max(im["top"], 0)
        covered += max(0.0, w) * max(0.0, h)
    return covered / area


def _ocr_reason(page, text: str, min_letters: int) -> str | None:
    """Why this page needs OCR, or None. Page numbers alone don't count as a text layer."""
    letters = sum(c.isalpha() for c in text)
    if letters >= OCR_RICH_TEXT_CHARS:
        return None
    if letters < min_letters:
        return "no usable text layer"
    if _image_coverage(page) >= OCR_IMAGE_COVERAGE:
        return "mostly an image with little text"
    return None


def _ocr_if_needed(page, number: int, text: str, name: str, done: list[int]) -> list[str]:
    """OCR lines for this page, or [] when not needed, disabled, unavailable or failed."""
    settings = get_settings()
    reason = _ocr_reason(page, text, settings.ocr_min_text_chars)
    if reason is None:
        return []
    if not settings.ocr_enabled:
        logger.warning("page %d of %s has %s but OCR is disabled (OCR_ENABLED=false)", number, name, reason)
        return []
    if len(done) >= settings.ocr_max_pages:
        logger.warning("page %d of %s skipped: OCR page limit (%d) reached", number, name, settings.ocr_max_pages)
        return []
    if not ocr.is_available():
        logger.warning("page %d of %s has %s and OCR is not installed; page skipped", number, name, reason)
        return []
    try:
        lines = ocr.ocr_page(page, settings.ocr_dpi, settings.ocr_min_confidence)
    except Exception:
        logger.warning("OCR failed on page %d of %s; page skipped", number, name, exc_info=True)
        return []
    logger.info("page %d of %s OCR'd (%s): %d lines", number, name, reason, len(lines))
    done.append(number)
    return lines


def _parse_pdf(file_path: str) -> str:
    """
    Text PDFs with tables. pdfplumber finds tables and we render each as a
    Markdown table (columns preserved); the remaining page text is extracted
    with table areas excluded so numbers aren't duplicated. Output carries
    page and table markers for the chunker. Scanned PDFs (no text layer) still
    yield nothing; there is no OCR.

    Falls back to pypdf plain text if pdfplumber is unavailable or fails.
    """
    name = Path(file_path).name
    try:
        import pdfplumber
    except ImportError:
        logger.warning(
            "pdfplumber is not installed: PDFs fall back to plain text with NO page numbers "
            "or table structure (every chunk section becomes 'unknown'). "
            "Fix: pip install pdfplumber (or re-run uv sync)."
        )
        return _parse_pdf_plain(file_path)

    try:
        parts: list[str] = []
        table_count = 0
        ocr_pages_done: list[int] = []  # page numbers that were OCR'd
        with pdfplumber.open(file_path) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.find_tables()
                    text_page = page
                    for t in tables:
                        # strict=False: a detected table box can extend slightly past the
                        # page edge, which would otherwise raise and lose the whole file.
                        text_page = text_page.outside_bbox(t.bbox, strict=False)
                    text = (text_page.extract_text() or "").strip()

                    block = [PAGE_MARK.format(n=number)]
                    if text:
                        block.append(text)
                    for t in tables:
                        md = _table_to_markdown(t.extract())
                        if md:
                            block += [TABLE_START, md, TABLE_END]
                            table_count += 1
                    ocr_lines = _ocr_if_needed(page, number, text, name, ocr_pages_done)
                    if ocr_lines:
                        block += [OCR_START, "\n".join(ocr_lines), OCR_END]
                except Exception:
                    # One bad page must not drop page numbers for the whole document.
                    logger.warning(
                        "pdfplumber failed on page %d of %s; using plain text for that page",
                        number, name, exc_info=True,
                    )
                    block = [PAGE_MARK.format(n=number), (page.extract_text() or "").strip()]
                parts.append("\n".join(block))
        logger.info(
            "pdf parsed with pdfplumber | file=%s pages=%d tables=%d ocr_pages=%s",
            name, len(parts), table_count, ocr_pages_done or "none",
        )
        return "\n\n".join(parts)
    except Exception:
        logger.warning(
            "pdfplumber could not open %s; falling back to plain text (no page numbers)",
            name, exc_info=True,
        )
        return _parse_pdf_plain(file_path)


def _parse_pdf_plain(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
