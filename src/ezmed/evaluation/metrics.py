"""Information retrieval metrics — Recall@k, MRR, NDCG@k, plus practical metrics."""


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    raise NotImplementedError


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    raise NotImplementedError


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    raise NotImplementedError
