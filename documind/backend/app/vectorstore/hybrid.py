"""
Hybrid retrieval helpers: keyword (BM25) search plus rank fusion.

Why: embeddings match meaning but are weak on exact tokens such as names and
numbers ("Priya", "54,500", "8.5%"). BM25 matches those exactly. The keyword
ranking is fused with the vector ranking by Reciprocal Rank Fusion (RRF),
which needs no score calibration between the two systems.

BM25 runs in-process over the requesting user's chunks only (tenant isolation
is kept by the backends' list_chunks(user_id)), so it behaves identically on
pgvector and Chroma. Cost is O(chunks per user) to build the index, so it is
cached per user and rebuilt after writes or after CACHE_TTL_SECONDS. Fine for
thousands of chunks per user; beyond that, move keyword search into the
database (Postgres full-text search).
"""
import math
import re
import time
from collections import Counter, defaultdict
from typing import Callable, Optional

CACHE_TTL_SECONDS = 60
RRF_K = 60

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or "
    "that the this to was what when where which who whom why will with does "
    "do did can could should would about into than then there their they "
    "you your we our me my".split()
)
_COMMA_IN_NUMBER = re.compile(r"(?<=\d),(?=\d)")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; '54,500' -> '54500' and '8.5' stays whole, so numbers match."""
    text = _COMMA_IN_NUMBER.sub("", text.lower())
    return [t for t in _TOKEN.findall(text) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tokens in enumerate(docs):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))

    def search(
        self, query_tokens: list[str], top_k: int, allowed: Optional[set[int]] = None
    ) -> list[tuple[int, float]]:
        """allowed: if given, only these document indexes may be returned (idf still
        uses the whole corpus, which is fine for ranking within the subset)."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for idx, tf in posting:
                if allowed is not None and idx not in allowed:
                    continue
                norm = 1 - self.b + self.b * self.doc_len[idx] / (self.avgdl or 1)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


# user_id -> (built_at, chunks, index)
_cache: dict[str, tuple[float, list[dict], BM25Index]] = {}


def invalidate(user_id: Optional[str] = None) -> None:
    """Drop cached indexes after writes. No user_id clears everyone."""
    if user_id is None:
        _cache.clear()
    else:
        _cache.pop(str(user_id), None)


def keyword_search(
    list_chunks: Callable[[str], list[dict]],
    user_id,
    query_text: str,
    top_k: int,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
) -> list[dict]:
    """
    Top-k chunks by BM25 for this user. Chunk dicts match the vector search shape.
    ts_from (inclusive) / ts_to (exclusive): epoch-second bounds on each chunk's
    'uploaded_ts'. A chunk with no upload time is excluded when a bound is set.
    """
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return []

    key = str(user_id)
    cached = _cache.get(key)
    if cached is None or time.monotonic() - cached[0] > CACHE_TTL_SECONDS:
        chunks = list_chunks(key)
        index = BM25Index([tokenize(c["content"]) for c in chunks])
        cached = (time.monotonic(), chunks, index)
        _cache[key] = cached

    _, chunks, index = cached
    allowed = None
    if ts_from is not None or ts_to is not None:
        allowed = {
            i
            for i, c in enumerate(chunks)
            if c.get("uploaded_ts") is not None
            and (ts_from is None or c["uploaded_ts"] >= ts_from)
            and (ts_to is None or c["uploaded_ts"] < ts_to)
        }
    return [chunks[i] for i, _ in index.search(query_tokens, top_k, allowed)]


def fuse(vector_hits: list[dict], keyword_hits: list[dict], top_k: int) -> list[dict]:
    """
    Reciprocal Rank Fusion of two ranked lists. Each result is tagged
    retrieved_via = 'vector' | 'keyword' | 'both' for the agent trace.
    """
    scores: dict[str, float] = defaultdict(float)
    via: dict[str, set[str]] = defaultdict(set)
    by_id: dict[str, dict] = {}

    for source, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            cid = hit["chunk_id"]
            scores[cid] += 1.0 / (RRF_K + rank)
            via[cid].add(source)
            by_id.setdefault(cid, hit)

    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    return [
        {**by_id[cid], "retrieved_via": "both" if len(via[cid]) == 2 else next(iter(via[cid]))}
        for cid in ordered
    ]
