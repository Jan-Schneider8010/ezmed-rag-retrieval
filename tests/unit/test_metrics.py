from math import isclose, log2

from ezmed.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_overlap_in_top_k() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "z"}
    assert recall_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 5) == 2 / 3


def test_recall_returns_zero_when_no_relevant_defined() -> None:
    assert recall_at_k(["a", "b"], set(), 5) == 0.0


def test_recall_at_k_perfect_score() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_recall_unaffected_by_items_beyond_k() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "e"}
    assert recall_at_k(retrieved, relevant, 3) == 0.5


def test_reciprocal_rank_first_hit_at_position_one() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_first_hit_at_position_three() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3


def test_reciprocal_rank_no_hit_returns_zero() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_reciprocal_rank_no_relevant_returns_zero() -> None:
    assert reciprocal_rank(["a", "b"], set()) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert ndcg_at_k(retrieved, relevant, 3) == 1.0


def test_ndcg_no_relevant_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


def test_ndcg_matches_known_example() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    expected = (1.0 + 1.0 / log2(4)) / (1.0 + 1.0 / log2(3))
    assert isclose(ndcg_at_k(retrieved, relevant, 3), expected)


def test_ndcg_handles_k_smaller_than_relevant() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c", "d", "e"}
    assert ndcg_at_k(retrieved, relevant, 2) == 1.0
