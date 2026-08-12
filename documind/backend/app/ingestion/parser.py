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


def _parse_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
