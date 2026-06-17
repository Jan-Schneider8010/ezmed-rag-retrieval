"""Loader for the PLABA benchmark: maps question → gold abstracts onto our schema.

PLABA (Attal et al. 2023) ships as a nested dict keyed by query id; each query
holds its lay question plus the PMIDs of its associated gold abstracts. We treat
each abstract as a single-section ParsedArticle and the question→PMIDs map as the
retrieval gold standard (relevance at PMID granularity).
"""

import json
import logging
from pathlib import Path
from typing import Any

from ezmed.ingestion.chunking import chunk_article
from ezmed.schemas import Chunk, ParsedArticle, Section
from ezmed.settings import settings

logger = logging.getLogger(__name__)


def load_plaba(path: Path) -> tuple[dict[str, ParsedArticle], list[dict[str, Any]]]:
    """Return (pmid -> ParsedArticle, queries) from PLABA data.json.

    Each query dict has keys: qid, question, gold_pmids.
    """
    with path.open() as f:
        raw = json.load(f)

    articles: dict[str, ParsedArticle] = {}
    queries: list[dict[str, Any]] = []

    for qid, q_entry in raw.items():
        question = q_entry["question"]
        gold_pmids: list[str] = []
        for key, value in q_entry.items():
            if key in {"question", "question_type"}:
                continue
            pmid = key
            gold_pmids.append(pmid)
            if pmid not in articles:
                articles[pmid] = _abstract_to_article(pmid, value)
        queries.append({"qid": qid, "question": question, "gold_pmids": gold_pmids})

    logger.info("loaded PLABA: %d queries, %d abstracts", len(queries), len(articles))
    return articles, queries


def _abstract_to_article(pmid: str, entry: dict[str, Any]) -> ParsedArticle:
    text = " ".join(entry["abstract"].values())
    return ParsedArticle(
        pmid=pmid,
        title=entry.get("Title", ""),
        abstract=[Section(title="Abstract", text=text)],
    )


def chunk_all(articles: dict[str, ParsedArticle]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles.values():
        chunks.extend(
            chunk_article(article, settings.chunk_size, settings.chunk_overlap)
        )
    logger.info("chunked %d articles into %d chunks", len(articles), len(chunks))
    return chunks


def subsample(
    queries: list[dict[str, Any]],
    articles: dict[str, ParsedArticle],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, ParsedArticle]]:
    """Keep the first `limit` queries and only the abstracts they reference."""
    queries = queries[:limit]
    keep_pmids = {pmid for q in queries for pmid in q["gold_pmids"]}
    articles = {pmid: art for pmid, art in articles.items() if pmid in keep_pmids}
    logger.info("subsampled: %d queries, %d abstracts", len(queries), len(articles))
    return queries, articles


def collapse_to_pmids(chunk_ids: list[str]) -> list[str]:
    """Reduce chunk IDs to unique PMIDs preserving order of first occurrence.

    PLABA relevance is judged at PMID granularity, so the ranked chunk list is
    collapsed to its underlying articles before scoring.
    """
    seen: set[str] = set()
    out: list[str] = []
    for cid in chunk_ids:
        pmid = cid.split(":", 1)[0]
        if pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out
