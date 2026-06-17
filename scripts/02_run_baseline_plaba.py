"""Baseline retrieval on PLABA: chunk -> embed -> Qdrant -> top-k search."""

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean
from typing import Any

from ezmed.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from ezmed.ingestion.plaba import (
    chunk_all,
    collapse_to_pmids,
    load_plaba,
    subsample,
)
from ezmed.llm.client import LLMClient
from ezmed.logging import configure_logging
from ezmed.retrieval.searcher import Searcher
from ezmed.schemas import Chunk, EmbeddedChunk
from ezmed.settings import settings
from ezmed.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

PLABA_PATH = Path("data/plaba/data.json")
RESULTS_PATH = Path("results/plaba_baseline.json")
COLLECTION = "plaba_chunks_plain"
TOP_K = 10
EMBED_CACHE_DIR = Path("data/processed/embeddings_cache")


def embed_chunks(chunks: list[Chunk], client: LLMClient) -> list[EmbeddedChunk]:
    vectors = client.embed([c.content for c in chunks])
    return [EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vectors)]


def main() -> None:
    args = _parse_args()
    configure_logging()
    if not settings.azure_openai_key or not settings.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set")

    started = time.time()
    articles, queries = load_plaba(PLABA_PATH)
    if args.limit is not None:
        queries, articles = subsample(queries, articles, args.limit)
    chunks = chunk_all(articles)

    client = LLMClient(
        deployment=settings.azure_openai_embedding_deployment,
        api_key=settings.azure_openai_key,
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        cache_dir=EMBED_CACHE_DIR,
    )
    embedded = embed_chunks(chunks, client)
    dim = len(embedded[0].embedding)
    logger.info("embedding dim = %d", dim)

    store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=COLLECTION,
        dim=dim,
    )
    store.reset()
    store.upsert(embedded)
    logger.info("upserted %d points into %r", len(embedded), COLLECTION)

    searcher = Searcher(embedder=client, store=store)
    per_query, latencies = _run_queries(searcher, queries)
    metrics = _aggregate(per_query)

    payload = {
        "variant": "baseline",
        "dataset": "plaba",
        "config": {
            "embedding_model": settings.azure_openai_embedding_deployment,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": TOP_K,
            "collection": COLLECTION,
            "granularity": "pmid",
        },
        "n_queries": len(queries),
        "n_chunks": len(chunks),
        "n_articles": len(articles),
        "metrics": metrics,
        "latency_ms": {
            "mean": mean(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "per_query": per_query,
        "runtime_s": round(time.time() - started, 2),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", RESULTS_PATH)
    logger.info(
        "baseline: Recall@5=%.3f Recall@10=%.3f MRR=%.3f NDCG@10=%.3f",
        metrics["recall_at_5"],
        metrics["recall_at_10"],
        metrics["mrr"],
        metrics["ndcg_at_10"],
    )


def _run_queries(
    searcher: Searcher, queries: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[float]]:
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        hits = searcher.search(q["question"], top_k=TOP_K)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_pmids = collapse_to_pmids([cid for cid, _ in hits])
        relevant = set(q["gold_pmids"])
        per_query.append({
            "qid": q["qid"],
            "n_gold": len(relevant),
            "n_retrieved_pmids": len(retrieved_pmids),
            "recall_at_5": recall_at_k(retrieved_pmids, relevant, 5),
            "recall_at_10": recall_at_k(retrieved_pmids, relevant, 10),
            "mrr": reciprocal_rank(retrieved_pmids, relevant),
            "ndcg_at_10": ndcg_at_k(retrieved_pmids, relevant, 10),
            "latency_ms": round(latency_ms, 1),
        })
    return per_query, latencies


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    return {k: mean(q[k] for q in per_query) for k in keys}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Subsample to first N queries and their gold abstracts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
