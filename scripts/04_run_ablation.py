"""Step 4: run the four variants over the QA set and persist retrieval + pool.

For every question, retrieve top-k from the right collection (plain vs HQ) with the
original or rewritten query, for all four variants. Writes:
  - results/ezmed_retrieval.json   {qa_id: {variant: [chunk_id ranked]}}  (feeds step 5)
  - results/ezmed_pool.json        {qa_id: [chunk_id]}   pool for judging (feeds gold)
  - results/ezmed_rewritten_queries.json

Match --corpus-limit/--seed and --embedding-deployment to step 2 so chunk_ids and
the embedding space line up with the ingested collections.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from ezmed.evaluation.judge import POOL_DEPTH, build_pool
from ezmed.evaluation.qa_dataset import load_jsonl
from ezmed.ingestion.corpus import load_chunk_index
from ezmed.ingestion.pipeline import parallel_map
from ezmed.llm.client import LLMClient
from ezmed.logging import configure_logging
from ezmed.retrieval.query_rewriter import QueryRewriter
from ezmed.retrieval.searcher import Searcher
from ezmed.schemas import QAPair
from ezmed.settings import settings
from ezmed.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
QA_PATH = Path("data/qa_dataset/qa_questions.jsonl")
PLAIN_COLLECTION = "ezmed_chunks_plain"
HQ_COLLECTION = "ezmed_chunks_hq"
DEFAULT_EMBED_DEPLOYMENT = "text-embedding-3-small"
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
    if args.top_k < POOL_DEPTH:
        raise ValueError(f"--top-k ({args.top_k}) must be >= pool depth ({POOL_DEPTH})")

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    embed_deployment = args.embedding_deployment or DEFAULT_EMBED_DEPLOYMENT
    suffix = f"_{args.tag}" if args.tag else ""
    plain_collection = PLAIN_COLLECTION + suffix
    hq_collection = HQ_COLLECTION + suffix

    qa_pairs = load_jsonl(args.qa)
    _, by_pmid = load_chunk_index(RAW_DIR, limit=args.corpus_limit, seed=args.seed)
    logger.info("loaded %d questions", len(qa_pairs))

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
    [probe] = embed_client.embed(["__deployment_probe__"])
    dim = len(probe)

    searchers = {
        plain_collection: Searcher(embed_client, _store(plain_collection, dim)),
        hq_collection: Searcher(embed_client, _store(hq_collection, dim)),
    }

    rewritten = _rewrite_queries(qa_pairs, chat_client, args.workers, suffix)

    retrieval: dict[str, dict[str, list[str]]] = {qa.qa_id: {} for qa in qa_pairs}
    tasks = [
        (variant, use_hq, use_rewritten, qa)
        for variant, use_hq, use_rewritten in VARIANT_SPECS
        for qa in qa_pairs
    ]

    def _retrieve(task: tuple[str, bool, bool, QAPair]) -> list[str]:
        _variant, use_hq, use_rewritten, qa = task
        searcher = searchers[hq_collection if use_hq else plain_collection]
        query = rewritten[qa.qa_id] if use_rewritten else qa.question
        return [cid for cid, _ in searcher.search(query, top_k=args.top_k)]

    hits_per_task = parallel_map(_retrieve, tasks, args.workers, "retrieval")
    for (variant, _use_hq, _use_rewritten, qa), hits in zip(tasks, hits_per_task, strict=True):
        retrieval[qa.qa_id][variant] = hits
    logger.info("retrieved %d variant×question pairs", len(tasks))

    pools = {
        qa.qa_id: build_pool(qa, retrieval[qa.qa_id], by_pmid.get(qa.pmid, []), k=POOL_DEPTH)
        for qa in qa_pairs
    }
    pool_sizes = [len(p) for p in pools.values()]

    (RESULTS_DIR / f"ezmed_retrieval{suffix}.json").write_text(json.dumps(retrieval, indent=2))
    (RESULTS_DIR / f"ezmed_pool{suffix}.json").write_text(json.dumps(pools, indent=2))
    run_config = {
        "dataset": "ezmed",
        "tag": args.tag,
        "embedding_model": embed_deployment,
        "query_rewrite_model": settings.azure_openai_chat_deployment,
        "corpus_limit": args.corpus_limit,
        "seed": args.seed,
        "top_k": args.top_k,
        "n_questions": len(qa_pairs),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    (RESULTS_DIR / f"ezmed_run_config{suffix}.json").write_text(json.dumps(run_config, indent=2))
    logger.info(
        "wrote retrieval + pool for %d questions (pool avg %.1f, max %d) in %.1fs",
        len(qa_pairs),
        sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0,
        max(pool_sizes) if pool_sizes else 0,
        time.time() - started,
    )


def _rewrite_queries(
    qa_pairs: list[QAPair], chat_client: LLMClient, workers: int, suffix: str
) -> dict[str, str]:
    rewriter = QueryRewriter(chat_client)
    logger.info("rewriting %d queries (workers=%d)", len(qa_pairs), workers)
    rewrites = parallel_map(lambda qa: rewriter.rewrite(qa.question)[0], qa_pairs, workers, "QR")
    rewritten = {qa.qa_id: rw for qa, rw in zip(qa_pairs, rewrites, strict=True)}
    dump: dict[str, Any] = {
        qa.qa_id: {"original": qa.question, "rewritten": rewritten[qa.qa_id]} for qa in qa_pairs
    }
    (RESULTS_DIR / f"ezmed_rewritten_queries{suffix}.json").write_text(json.dumps(dump, indent=2))
    return rewritten


def _store(collection: str, dim: int) -> VectorStore:
    return VectorStore(
        host=settings.qdrant_host, port=settings.qdrant_port, collection=collection, dim=dim
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=QA_PATH, help=f"QA JSONL ({QA_PATH}).")
    parser.add_argument(
        "--corpus-limit", type=int, default=100, help="Match step 2 --limit. Default 100."
    )
    parser.add_argument("--seed", type=int, default=42, help="Match step 2 --seed. Default 42.")
    parser.add_argument(
        "--embedding-deployment",
        default=None,
        help=f"Must match step 2 (default {DEFAULT_EMBED_DEPLOYMENT}).",
    )
    parser.add_argument("--tag", default="", help="Collection/result suffix.")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Retrieval depth; must be >= pool depth (10)."
    )
    parser.add_argument(
        "--workers", type=int, default=64, help="Concurrent query-rewrite + retrieval calls."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
