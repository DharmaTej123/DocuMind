"""
Central logging setup. Called first thing in app/main.py, before any other app
module is imported, so import-time messages (e.g. the graph diagram) are kept.

force=True replaces handlers that libraries may have attached to the root
logger already; plain basicConfig() silently does nothing in that case, which
is why app logs went missing under uvicorn.
"""
import logging
import sys

NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "hpack", "chromadb", "opentelemetry",
    "posthog", "multipart", "pdfminer", "pdfplumber", "PIL",
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
