"""Ingestion building blocks: chunks -> (HQ) -> embeddings -> Qdrant.

Two-collection rule (see CLAUDE.md): the corpus is embedded twice, never four
times. `build_plain_collection` feeds `baseline`/`qr_only`; `build_hq_collection`
feeds `hq_only`/`both`. Shared by `scripts/02_ingest.py` and
`scripts/04_run_corpus_ablation.py`. The frozen Stage-1 PLABA script keeps its
own private copies on purpose."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from ezmed.ingestion.enrichment import HypotheticalQuestionGenerator, enriched_text
from ezmed.llm.client import LLMClient
from ezmed.schemas import Chunk, EmbeddedChunk
from ezmed.settings import settings
from ezmed.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
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


def build_plain_collection(
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


def build_hq_collection(
    chunks: list[Chunk],
    embed_client: LLMClient,
    chat_client: LLMClient,
    collection: str,
    workers: int,
) -> tuple[VectorStore, list[Chunk]]:
    """Enrich chunks with HQ, embed content+HQ, upsert. Returns (store, enriched)."""
    hq_gen = HypotheticalQuestionGenerator(chat_client, settings.hq_per_chunk)
    logger.info(
        "generating HQ for %d chunks (k=%d, workers=%d)",
        len(chunks), settings.hq_per_chunk, workers,
    )
    enriched = parallel_map(hq_gen.enrich, chunks, workers, "HQ enrich")

    vectors = embed_client.embed([enriched_text(c) for c in enriched])
    embedded = [
        EmbeddedChunk(chunk=c, embedding=v)
        for c, v in zip(enriched, vectors, strict=True)
    ]
    store = _store(collection, len(embedded[0].embedding))
    store.reset()
    store.upsert(embedded)
    logger.info("upserted %d points into %r", len(embedded), collection)
    return store, enriched


def _store(collection: str, dim: int) -> VectorStore:
    return VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=collection,
        dim=dim,
    )
