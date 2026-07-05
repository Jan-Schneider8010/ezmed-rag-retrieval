"""Step 2: ingest the self-built cardiology corpus into two Qdrant collections.

Full-text papers from data/raw/ -> chunks -> embeddings, plus an HQ-enriched copy:
  - ezmed_chunks_plain   (baseline, qr_only)
  - ezmed_chunks_hq      (hq_only, both)

Embeds with text-embedding-3-small by default (Stage-1 finding: small ~= large,
~80% cheaper). HQ generation runs on the frozen gpt-4.1-mini. Both are real money
on the full corpus (~100 papers x ~20 chunks x 2) — check --limit before a full run.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from ezmed.ingestion.corpus import load_corpus
from ezmed.ingestion.pipeline import build_hq_collection, build_plain_collection
from ezmed.ingestion.plaba import chunk_all
from ezmed.llm.client import LLMClient
from ezmed.logging import configure_logging
from ezmed.schemas import Chunk
from ezmed.settings import settings

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
PLAIN_COLLECTION = "ezmed_chunks_plain"
HQ_COLLECTION = "ezmed_chunks_hq"
DEFAULT_EMBED_DEPLOYMENT = "text-embedding-3-small"
EMBED_CACHE_DIR = Path("data/processed/embeddings_cache")
CHAT_CACHE_DIR = Path("data/processed/completions_cache")


def main() -> None:
    args = _parse_args()
    configure_logging()
    if not settings.azure_openai_key or not settings.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set")

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    embed_deployment = args.embedding_deployment or DEFAULT_EMBED_DEPLOYMENT
    suffix = f"_{args.tag}" if args.tag else ""
    plain_collection = PLAIN_COLLECTION + suffix
    hq_collection = HQ_COLLECTION + suffix

    articles = load_corpus(RAW_DIR, limit=args.limit, seed=args.seed)
    chunks = chunk_all(articles)
    if not chunks:
        raise RuntimeError("no chunks produced from corpus")

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

    if not args.hq_only:
        build_plain_collection(chunks, embed_client, plain_collection)
    if not args.plain_only:
        _, enriched = build_hq_collection(
            chunks, embed_client, chat_client, hq_collection, args.workers
        )
        _dump_hq_questions(enriched, suffix)

    logger.info(
        "ingested %d articles / %d chunks in %.1fs (embed=%s)",
        len(articles),
        len(chunks),
        time.time() - started,
        embed_deployment,
    )


def _dump_hq_questions(enriched: list[Chunk], suffix: str) -> None:
    dump = {c.chunk_id: {"content": c.content, "hq_questions": c.hq_questions} for c in enriched}
    path = RESULTS_DIR / f"ezmed_hq_questions{suffix}.json"
    path.write_text(json.dumps(dump, indent=2))
    logger.info("dumped HQ questions for %d chunks to %s", len(enriched), path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of papers to ingest (deterministic subsample). Default 100.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Subsample seed (default 42).")
    parser.add_argument(
        "--embedding-deployment",
        default=None,
        help=f"Azure embedding deployment (default {DEFAULT_EMBED_DEPLOYMENT}).",
    )
    parser.add_argument("--tag", default="", help="Suffix for collections + result files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Concurrent LLM calls for HQ generation.",
    )
    parser.add_argument(
        "--plain-only", action="store_true", help="Build only the plain collection."
    )
    parser.add_argument("--hq-only", action="store_true", help="Build only the HQ collection.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
