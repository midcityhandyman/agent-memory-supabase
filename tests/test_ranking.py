from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.ranking import (Candidate, RankingConfig, blend_score, helped_rate,
                                  outcome_multiplier, rank_candidates, recency_factor,
                                  relevance)

NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def test_relevance_is_weighted_mean():
    assert relevance(1.0, 1.0) == pytest.approx(1.0)
    assert relevance(1.0, 0.0) == pytest.approx(0.4)
    assert relevance(0.0, 1.0) == pytest.approx(0.6)
    assert relevance(0.5, 0.5) == pytest.approx(0.5)


def test_relevance_clamps_and_handles_none():
    assert relevance(5.0, -3.0) == pytest.approx(0.4)
    assert relevance(None, None) == 0.0


def test_relevance_with_zero_weights_is_zero():
    assert relevance(1.0, 1.0, RankingConfig(w_text=0.0, w_vector=0.0)) == 0.0


def test_weights_are_normalised():
    a = relevance(0.8, 0.2, RankingConfig(w_text=0.4, w_vector=0.6))
    b = relevance(0.8, 0.2, RankingConfig(w_text=4.0, w_vector=6.0))
    assert a == pytest.approx(b)


def test_no_feedback_is_neutral():
    assert helped_rate(0, 0) == pytest.approx(0.5)
    assert outcome_multiplier(0, 0) == pytest.approx(1.0)


def test_balanced_feedback_is_neutral():
    assert outcome_multiplier(7, 7) == pytest.approx(1.0)


def test_one_vote_moves_the_needle_only_a_little():
    assert outcome_multiplier(1, 0) == pytest.approx(1.1)
    assert outcome_multiplier(0, 1) == pytest.approx(0.9)


def test_multiplier_is_monotonic_and_bounded():
    previous = 0.0
    for helped in range(0, 200, 5):
        value = outcome_multiplier(helped, 3)
        assert value > previous
        assert 0.5 < value < 1.5
        previous = value
    assert outcome_multiplier(0, 10_000) > 0.5
    assert outcome_multiplier(10_000, 0) < 1.5


def test_negative_counts_are_treated_as_zero():
    assert outcome_multiplier(-4, -2) == pytest.approx(1.0)


def test_outcome_weight_zero_disables_reranking():
    cfg = RankingConfig(outcome_weight=0.0)
    assert outcome_multiplier(50, 0, cfg) == pytest.approx(1.0)
    assert outcome_multiplier(0, 50, cfg) == pytest.approx(1.0)


def test_recency_factor_shape():
    assert recency_factor(0) == pytest.approx(1.0)
    assert recency_factor(90) == pytest.approx(0.85 + 0.15 * 0.5)
    assert recency_factor(100_000) == pytest.approx(0.85)
    assert recency_factor(-10) == pytest.approx(1.0)
    assert recency_factor(None) == pytest.approx(1.0)


def test_blend_score_is_product_of_parts():
    score = blend_score(0.6, 0.8, helped=3, not_helped=1, age_days=45)
    expected = relevance(0.6, 0.8) * outcome_multiplier(3, 1) * recency_factor(45)
    assert score == pytest.approx(expected)


def test_blend_score_known_value():
    # relevance 0.72, multiplier 1 + 0.5 * (2 * 5/8 - 1) = 1.125, recency 1.0
    assert blend_score(0.6, 0.8, helped=3, not_helped=1, age_days=0) == pytest.approx(0.81)


def test_helpful_memory_overtakes_a_slightly_more_similar_one():
    similar_but_unhelpful = Candidate("a", NOW, text_score=0.5, vector_score=0.80,
                                      helped=0, not_helped=4)
    helpful = Candidate("b", NOW, text_score=0.5, vector_score=0.70, helped=4, not_helped=0)
    ranked = rank_candidates([similar_but_unhelpful, helpful], now=NOW)
    assert [r.candidate.id for r in ranked] == ["b", "a"]


def test_feedback_cannot_rescue_an_irrelevant_memory():
    irrelevant = Candidate("a", NOW, text_score=0.0, vector_score=0.05, helped=500)
    relevant = Candidate("b", NOW, text_score=0.6, vector_score=0.7, not_helped=3)
    ranked = rank_candidates([irrelevant, relevant], now=NOW)
    assert ranked[0].candidate.id == "b"


def test_ties_break_by_newest_then_id():
    old = Candidate("a", NOW - timedelta(seconds=1), vector_score=0.5)
    new_b = Candidate("b", NOW, vector_score=0.5)
    new_c = Candidate("c", NOW, vector_score=0.5)
    cfg = RankingConfig(recency_weight=0.0)
    ranked = rank_candidates([old, new_c, new_b], now=NOW, config=cfg)
    assert [r.candidate.id for r in ranked] == ["b", "c", "a"]


def test_limit_is_applied_after_sorting():
    cands = [Candidate(str(i), NOW, vector_score=i / 10) for i in range(10)]
    ranked = rank_candidates(cands, now=NOW, limit=3)
    assert [r.candidate.id for r in ranked] == ["9", "8", "7"]
    assert rank_candidates(cands, now=NOW, limit=0) == []


def test_without_vector_puts_all_weight_on_text():
    cfg = RankingConfig().without_vector()
    assert relevance(0.5, 0.0, cfg) == pytest.approx(0.5)
