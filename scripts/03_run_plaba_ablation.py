"""Full 2x2 ablation on PLABA: baseline, qr_only, hq_only, both.

Two collections only: `plaba_chunks_plain` (baseline + qr_only) and
`plaba_chunks_hq` (hq_only + both). Query rewriting is a runtime transform applied
in front of either collection — it gets no collection of its own.

Writes one `results/plaba_<variant>.json` per variant (same schema as the Sprint-1
baseline), the cross-variant `results/plaba_ablation.json` (means + pairwise
Wilcoxon/Bonferroni), and human-readable artefact dumps of the generated HQ
questions and rewritten queries.

Content generation (HQ enrichment, query rewriting) is the bottleneck — one LLM
call per chunk/query — so those run concurrently via a thread pool. Embeddings are
already batched server-side. Use --tag / --embedding-deployment to run an alternate
embedding model side by side without clobbering the main results.
"""

import argparse
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, TypeVar

from ezmed.evaluation.analysis import build_plaba_ablation
from ezmed.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from ezmed.ingestion.enrichment import HypotheticalQuestionGenerator, enriched_text
from ezmed.ingestion.plaba import (
    chunk_all,
    collapse_to_pmids,
    load_plaba,
    subsample,
)
from ezmed.llm.client import LLMClient
from ezmed.logging import configure_logging
from ezmed.retrieval.query_rewriter import QueryRewriter
from ezmed.retrieval.searcher import Searcher
from ezmed.schemas import Chunk, EmbeddedChunk
from ezmed.settings import settings
from ezmed.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

PLABA_PATH = Path("data/plaba/data.json")
RESULTS_DIR = Path("results")
PLAIN_COLLECTION = "plaba_chunks_plain"
HQ_COLLECTION = "plaba_chunks_hq"
TOP_K = 10
EMBED_CACHE_DIR = Path("data/processed/embeddings_cache")
CHAT_CACHE_DIR = Path("data/processed/completions_cache")

# (variant, use_hq_collection, use_rewritten_query)
VARIANT_SPECS = [
    ("baseline", False, False),
    ("qr_only", False, True),
    ("hq_only", True, False),
    ("both", True, True),
]


def main() -> None:
    args = _parse_args()
    configure_logging()
    if not settings.azure_openai_key or not settings.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set")

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    embed_deployment = args.embedding_deployment or settings.azure_openai_embedding_deployment
    suffix = f"_{args.tag}" if args.tag else ""
    plain_collection = PLAIN_COLLECTION + suffix
    hq_collection = HQ_COLLECTION + suffix

    articles, queries = load_plaba(PLABA_PATH)
    if args.limit is not None:
        queries, articles = subsample(queries, articles, args.limit)
    chunks = chunk_all(articles)

    embed_client = LLMClient(
        deployment=embed_deployment,
        api_key=settings.azure_openai_key,
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        cache_dir=EMBED_CACHE_DIR,
    )
    chat_client = LLMClient(
        deployment=settings.azure_openai_chat_deployment,
        api_key=settings.azure_openai_key,
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        cache_dir=CHAT_CACHE_DIR,
    )

    # Fail fast on a bad embedding deployment before any expensive generation.
    embed_client.embed(["__deployment_probe__"])
    logger.info("embedding deployment %r reachable", embed_deployment)

    store_plain = _build_plain_collection(chunks, embed_client, plain_collection)
    store_hq = _build_hq_collection(
        chunks, embed_client, chat_client, hq_collection, args.workers, suffix
    )
    rewritten = _rewrite_queries(queries, chat_client, args.workers, suffix)

    searchers = {
        plain_collection: Searcher(embedder=embed_client, store=store_plain),
        hq_collection: Searcher(embedder=embed_client, store=store_hq),
    }

    per_variant: dict[str, list[dict[str, Any]]] = {}
    for variant, use_hq, use_rewritten in VARIANT_SPECS:
        collection = hq_collection if use_hq else plain_collection
        searcher = searchers[collection]

        def query_for(q: dict[str, Any], _rw: bool = use_rewritten) -> str:
            return rewritten[q["qid"]] if _rw else q["question"]

        per_query, latencies = _run_variant(searcher, queries, query_for)
        per_variant[variant] = per_query
        metrics = _aggregate(per_query)
        _write_variant_result(
            variant, collection, embed_deployment, queries, chunks, articles,
            metrics, per_query, latencies, use_rewritten, suffix,
        )
        logger.info(
            "%s: Recall@5=%.3f Recall@10=%.3f MRR=%.3f NDCG@10=%.3f",
            variant,
            metrics["recall_at_5"],
            metrics["recall_at_10"],
            metrics["mrr"],
            metrics["ndcg_at_10"],
        )

    ablation = build_plaba_ablation(per_variant)
    ablation["embedding_model"] = embed_deployment
    out = RESULTS_DIR / f"plaba_ablation{suffix}.json"
    out.write_text(json.dumps(ablation, indent=2))
    logger.info("wrote %s (%.1fs total)", out, time.time() - started)


def _parallel_map(
    fn: Callable[[T], R], items: list[T], workers: int, desc: str
) -> list[R]:
    """Run fn over items concurrently, preserving input order. Network-bound work."""
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    results: list[R] = [None] * len(items)  # type: ignore[list-item]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if done % 200 == 0 or done == len(items):
                logger.info("%s: %d/%d", desc, done, len(items))
    return results


def _build_plain_collection(
    chunks: list[Chunk], embed_client: LLMClient, collection: str
) -> VectorStore:
    vectors = embed_client.embed([c.content for c in chunks])
    embedded = [
        EmbeddedChunk(chunk=c, embedding=v)
        for c, v in zip(chunks, vectors, strict=True)
    ]
    store = _store(collection, len(embedded[0].embedding))
    store.reset()
    store.upsert(embedded)
    logger.info("upserted %d points into %r", len(embedded), collection)
    return store


def _build_hq_collection(
    chunks: list[Chunk],
    embed_client: LLMClient,
    chat_client: LLMClient,
    collection: str,
    workers: int,
    suffix: str,
) -> VectorStore:
    hq_gen = HypotheticalQuestionGenerator(chat_client, settings.hq_per_chunk)
    logger.info(
        "generating HQ for %d chunks (k=%d, workers=%d)",
        len(chunks), settings.hq_per_chunk, workers,
    )
    enriched = _parallel_map(hq_gen.enrich, chunks, workers, "HQ enrich")
    _dump_hq_questions(enriched, suffix)

    vectors = embed_client.embed([enriched_text(c) for c in enriched])
    embedded = [
        EmbeddedChunk(chunk=c, embedding=v)
        for c, v in zip(enriched, vectors, strict=True)
    ]
    store = _store(collection, len(embedded[0].embedding))
    store.reset()
    store.upsert(embedded)
    logger.info("upserted %d points into %r", len(embedded), collection)
    return store


def _rewrite_queries(
    queries: list[dict[str, Any]], chat_client: LLMClient, workers: int, suffix: str
) -> dict[str, str]:
    rewriter = QueryRewriter(chat_client)
    logger.info("rewriting %d queries (workers=%d)", len(queries), workers)
    rewrites = _parallel_map(
        lambda q: rewriter.rewrite(q["question"])[0], queries, workers, "QR"
    )
    rewritten = {q["qid"]: rw for q, rw in zip(queries, rewrites, strict=True)}
    dump = {
        q["qid"]: {"original": q["question"], "rewritten": rewritten[q["qid"]]}
        for q in queries
    }
    (RESULTS_DIR / f"plaba_rewritten_queries{suffix}.json").write_text(
        json.dumps(dump, indent=2)
    )
    return rewritten


def _run_variant(
    searcher: Searcher,
    queries: list[dict[str, Any]],
    query_for: Callable[[dict[str, Any]], str],
) -> tuple[list[dict[str, Any]], list[float]]:
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        hits = searcher.search(query_for(q), top_k=TOP_K)
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


def _write_variant_result(
    variant: str,
    collection: str,
    embed_deployment: str,
    queries: list[dict[str, Any]],
    chunks: list[Chunk],
    articles: dict[str, Any],
    metrics: dict[str, float],
    per_query: list[dict[str, Any]],
    latencies: list[float],
    query_rewriting: bool,
    suffix: str,
) -> None:
    payload = {
        "variant": variant,
        "dataset": "plaba",
        "config": {
            "embedding_model": embed_deployment,
            "chat_model": settings.azure_openai_chat_deployment,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "hq_per_chunk": settings.hq_per_chunk,
            "query_rewriting": query_rewriting,
            "top_k": TOP_K,
            "collection": collection,
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
    }
    path = RESULTS_DIR / f"plaba_{variant}{suffix}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", path)


def _dump_hq_questions(enriched: list[Chunk], suffix: str) -> None:
    dump = {
        c.chunk_id: {"content": c.content, "hq_questions": c.hq_questions}
        for c in enriched
    }
    (RESULTS_DIR / f"plaba_hq_questions{suffix}.json").write_text(json.dumps(dump, indent=2))
    logger.info("dumped HQ questions for %d chunks", len(enriched))


def _store(collection: str, dim: int) -> VectorStore:
    return VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=collection,
        dim=dim,
    )


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    return {k: mean(q[k] for q in per_query) for k in keys}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Subsample to first N queries and their gold abstracts (cheap test run).",
    )
    parser.add_argument(
        "--embedding-deployment",
        default=None,
        help="Azure embedding deployment to use (default: settings value).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix for collections + result files, e.g. 'small' for an alternate "
             "embedding model. Empty = main results.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent LLM calls for HQ generation and query rewriting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
