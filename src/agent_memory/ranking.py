"""Pure Python reference implementation of the recall ranking.

The arithmetic here mirrors the SQL function public.memory_blend_score in
sql/003_recall.sql term for term, and rank_candidates mirrors the final
select of public.recall_memories. Keeping a reference in plain Python means
the ranking can be unit tested, tuned and explained with no database.

Score = relevance * outcome_multiplier * recency_factor

  relevance           weighted mean of text and vector similarity, each in [0, 1]
  outcome_multiplier  smoothed "did this memory help" rate, centred on 1
  recency_factor      mild exponential decay with a floor
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Iterable, Optional


@dataclass(frozen=True)
class RankingConfig:
    """Weights for the blend. Defaults match the SQL function defaults."""

    w_text: float = 0.4
    w_vector: float = 0.6
    outcome_weight: float = 0.5
    prior_helped: float = 2.0
    prior_not_helped: float = 2.0
    recency_weight: float = 0.15
    half_life_days: float = 90.0

    def __post_init__(self) -> None:
        # Outside these ranges the score stops meaning anything: a zero half
        # life or zero priors divide by zero, and an outcome weight above 1
        # lets enough bad feedback turn a score negative.
        if self.w_text < 0 or self.w_vector < 0:
            raise ValueError("w_text and w_vector must not be negative")
        if not 0 <= self.outcome_weight <= 1:
            raise ValueError("outcome_weight must be between 0 and 1")
        if not 0 <= self.recency_weight <= 1:
            raise ValueError("recency_weight must be between 0 and 1")
        if self.prior_helped <= 0 or self.prior_not_helped <= 0:
            raise ValueError("prior_helped and prior_not_helped must be positive")
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive")

    def without_vector(self) -> "RankingConfig":
        """Config used when the query has no embedding: text carries it all."""
        return replace(self, w_vector=0.0)


DEFAULT_CONFIG = RankingConfig()


def _clamp01(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def relevance(text_score: Optional[float], vector_score: Optional[float],
              config: RankingConfig = DEFAULT_CONFIG) -> float:
    """Weighted mean of the two similarity signals. Zero if both weights are zero."""
    total = config.w_text + config.w_vector
    if total == 0:
        return 0.0
    blended = config.w_text * _clamp01(text_score) + config.w_vector * _clamp01(vector_score)
    return blended / total


def helped_rate(helped: int, not_helped: int,
                config: RankingConfig = DEFAULT_CONFIG) -> float:
    """Posterior mean of a Beta prior updated with the recorded outcomes.

    With the default 2 and 2 prior a memory with no feedback sits at 0.5, one
    lucky vote moves it only to 0.6, and it takes a real track record to get
    near 0 or 1. Neutral outcomes are not counted either way.
    """
    h = max(int(helped), 0)
    n = max(int(not_helped), 0)
    return (h + config.prior_helped) / (h + n + config.prior_helped + config.prior_not_helped)


def outcome_multiplier(helped: int, not_helped: int,
                       config: RankingConfig = DEFAULT_CONFIG) -> float:
    """Map the helped rate onto [1 - outcome_weight, 1 + outcome_weight]."""
    return 1.0 + config.outcome_weight * (2.0 * helped_rate(helped, not_helped, config) - 1.0)


def recency_factor(age_days: Optional[float],
                   config: RankingConfig = DEFAULT_CONFIG) -> float:
    """1.0 for a brand new memory, decaying toward 1 - recency_weight."""
    age = max(float(age_days or 0.0), 0.0)
    decay = 0.5 ** (age / config.half_life_days)
    return (1.0 - config.recency_weight) + config.recency_weight * decay


def blend_score(text_score: Optional[float], vector_score: Optional[float],
                helped: int = 0, not_helped: int = 0, age_days: Optional[float] = 0.0,
                config: RankingConfig = DEFAULT_CONFIG) -> float:
    """Final score for one candidate. Mirrors public.memory_blend_score."""
    return (
        relevance(text_score, vector_score, config)
        * outcome_multiplier(helped, not_helped, config)
        * recency_factor(age_days, config)
    )


@dataclass(frozen=True)
class Candidate:
    """Everything the ranker needs to know about one retrieved memory."""

    id: str
    created_at: datetime
    text_score: float = 0.0
    vector_score: float = 0.0
    helped: int = 0
    not_helped: int = 0


@dataclass(frozen=True)
class Ranked:
    candidate: Candidate
    score: float


def rank_candidates(candidates: Iterable[Candidate], now: datetime,
                    config: RankingConfig = DEFAULT_CONFIG,
                    limit: Optional[int] = None) -> list[Ranked]:
    """Score and order candidates the way recall_memories does.

    Order: score descending, then newest first, then id ascending, so the
    result is fully deterministic.
    """
    ranked = []
    for cand in candidates:
        age_days = (now - cand.created_at).total_seconds() / 86400.0
        score = blend_score(cand.text_score, cand.vector_score, cand.helped,
                            cand.not_helped, age_days, config)
        ranked.append(Ranked(cand, score))
    ranked.sort(key=lambda r: r.candidate.id)
    ranked.sort(key=lambda r: r.candidate.created_at, reverse=True)
    ranked.sort(key=lambda r: r.score, reverse=True)
    if limit is not None:
        ranked = ranked[: max(limit, 0)]
    return ranked
