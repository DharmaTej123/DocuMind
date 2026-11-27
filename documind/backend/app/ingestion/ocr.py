"""
OCR for scanned PDF pages, using RapidOCR (ONNX, CPU). The models ship inside the
package: nothing is downloaded and no page image leaves the machine.

RapidOCR returns scattered text boxes, not rows. rows_from_detections() rebuilds
reading order: boxes on the same visual line are joined into one row, and a wide
horizontal gap between boxes becomes ' | ', so a scanned table row stays readable
as 'Date | Description | Branch | Venue'. It does not recover merged cells or
borders; blank cells simply produce fewer ' | ' separators.
"""
import logging
import time
from functools import lru_cache
from statistics import median

logger = logging.getLogger("documind.ocr")

GAP_FACTOR = 1.5  # a gap wider than this many text heights starts a new cell


@lru_cache
def get_engine():
    """The shared OCR engine, or None if RapidOCR is not installed (logged once)."""
    # RapidOCR logs model-loading chatter at INFO through its own logger.
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    try:
        from rapidocr import RapidOCR

        return RapidOCR()
    except Exception:
        logger.warning(
            "OCR is unavailable, so scanned PDF pages will be skipped. "
            "Fix: pip install rapidocr onnxruntime (or re-run uv sync).",
            exc_info=True,
        )
        return None


def is_available() -> bool:
    return get_engine() is not None


def rows_from_detections(boxes, texts, scores, min_confidence: float = 0.5) -> list[str]:
    """Group detected text boxes into reading-order lines. Pure function (no OCR needed)."""
    items = []
    for box, text, score in zip(boxes, texts, scores):
        text = " ".join(str(text).split())
        if not text or float(score) < min_confidence:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        items.append(
            {"text": text, "x0": min(xs), "x1": max(xs), "cy": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys)}
        )
    if not items:
        return []

    tolerance = 0.5 * median(i["h"] for i in items)
    items.sort(key=lambda i: i["cy"])
    rows, current = [], [items[0]]
    for item in items[1:]:
        row_cy = sum(i["cy"] for i in current) / len(current)
        if abs(item["cy"] - row_cy) <= tolerance:
            current.append(item)
        else:
            rows.append(current)
            current = [item]
    rows.append(current)

    lines = []
    for row in rows:
        row.sort(key=lambda i: i["x0"])
        height = median(i["h"] for i in row)
        line = row[0]["text"]
        for prev, nxt in zip(row, row[1:]):
            line += (" | " if nxt["x0"] - prev["x1"] > GAP_FACTOR * height else " ") + nxt["text"]
        lines.append(line)
    return lines


def ocr_page(page, dpi: int = 200, min_confidence: float = 0.5) -> list[str]:
    """
    OCR one pdfplumber page. Returns text lines ([] if nothing was read).
    Raises RuntimeError if the engine is unavailable.
    """
    engine = get_engine()
    if engine is None:
        raise RuntimeError("OCR engine unavailable")

    import numpy as np

    started = time.perf_counter()
    image = page.to_image(resolution=dpi).original.convert("RGB")
    result = engine(np.array(image))

    texts = getattr(result, "txts", None)
    if not texts:
        logger.info("ocr | page read no text (%.0f ms)", (time.perf_counter() - started) * 1000)
        return []
    lines = rows_from_detections(result.boxes, texts, result.scores, min_confidence)
    logger.info(
        "ocr | boxes=%d kept_lines=%d dpi=%d %.0f ms",
        len(texts), len(lines), dpi, (time.perf_counter() - started) * 1000,
    )
    return lines
