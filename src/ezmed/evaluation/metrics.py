"""Recall@k, MRR, NDCG@k. Granularity-agnostic over the ID space."""

from math import log2


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


mean_reciprocal_rank = reciprocal_rank


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / log2(i + 2)
        for i, item in enumerate(retrieved[:k])
        if item in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
