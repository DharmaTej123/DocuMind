"""
Upload-date helpers for filtering. Everything is UTC. Documents store
created_at as a naive UTC datetime (datetime.utcnow), and Chroma stores the same
instant as integer epoch seconds in 'uploaded_ts' (Chroma can only compare
numbers, not dates).

Filter semantics: uploaded_after and uploaded_before are both INCLUSIVE calendar
days, so the range is [start of uploaded_after, start of the day after uploaded_before).
"""
import calendar
from datetime import date, datetime, timedelta, timezone
from typing import Optional


def to_epoch(dt: datetime) -> int:
    """Epoch seconds for a datetime; naive values are treated as UTC."""
    return calendar.timegm(dt.utctimetuple())


def from_epoch(ts: int) -> datetime:
    """Naive UTC datetime, matching how Document.created_at is stored."""
    return datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None)


def bounds(
    date_from: Optional[str], date_to: Optional[str]
) -> tuple[Optional[int], Optional[int]]:
    """
    ISO 'YYYY-MM-DD' strings -> (start_ts inclusive, end_ts exclusive) epoch
    seconds, either may be None. Raises ValueError on a malformed date.
    """
    start = end = None
    if date_from:
        d = date.fromisoformat(date_from)
        start = to_epoch(datetime(d.year, d.month, d.day))
    if date_to:
        d = date.fromisoformat(date_to) + timedelta(days=1)
        end = to_epoch(datetime(d.year, d.month, d.day))
    return start, end
