"""A small full text scorer that stands in for Postgres ts_rank_cd offline.

This is an approximation, not a port. Postgres uses the Snowball English
stemmer and cover density ranking. Here we lowercase, drop stop words, strip
a few common suffixes, and score on query term coverage and term frequency,
with title matches weighted above body matches (as weights A and B are in
the search_tsv column). Like the SQL side with normalisation flag 32, the
result is squashed into [0, 1) with r / (r + 1).
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOP_WORDS = frozenset(
    """a an and are as at be but by for from had has have how i if in into is it its
    of on or our so that the their them then there these they this to was we were
    what when where which who will with you your do does did not no can should""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ingly", "edly", "ing", "ies", "ed", "es", "ly", "s")

TITLE_WEIGHT = 1.0  # weight A in the tsvector
BODY_WEIGHT = 0.4   # weight B in the tsvector


def stem(token: str) -> str:
    """Very light suffix stripping. Good enough to match plurals and tenses."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non alphanumerics, drop stop words, stem."""
    return [stem(tok) for tok in _TOKEN.findall((text or "").lower()) if tok not in STOP_WORDS]


def text_score(query: str, title: str, content: str) -> float:
    """Score in [0, 1). Zero when no query term appears in the document."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    title_counts = Counter(tokenize(title))
    body_counts = Counter(tokenize(content))
    matched = 0
    weighted_tf = 0.0
    for term in query_terms:
        hits = TITLE_WEIGHT * title_counts.get(term, 0) + BODY_WEIGHT * body_counts.get(term, 0)
        if hits > 0:
            matched += 1
            weighted_tf += 1.0 + math.log(1.0 + hits)
    if matched == 0:
        return 0.0
    coverage = matched / len(query_terms)
    raw = coverage * weighted_tf
    return raw / (raw + 1.0)


def matches(query: str, title: str, content: str) -> bool:
    """True when at least one query term appears (the OR semantics of recall)."""
    return text_score(query, title, content) > 0.0
