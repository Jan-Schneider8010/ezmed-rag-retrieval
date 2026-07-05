"""Load the self-built cardiology corpus from parsed PMC JATS XML on disk.

Stage-2 counterpart to `plaba.py`: `data/raw/*.xml` -> ParsedArticle. Malformed
files (missing <article>/pmid, broken XML) are a boundary condition on a
1000-file OA dump, so they are skipped and logged rather than aborting the run."""

import logging
import random
from pathlib import Path

from lxml import etree

from ezmed.ingestion.parser import parse_article
from ezmed.ingestion.plaba import chunk_all
from ezmed.schemas import ParsedArticle

logger = logging.getLogger(__name__)


def load_corpus(
    raw_dir: Path, limit: int | None = None, seed: int = 42
) -> dict[str, ParsedArticle]:
    """Return pmid -> ParsedArticle. With `limit`, take a deterministic subsample."""
    xml_paths = sorted(raw_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"no *.xml files in {raw_dir}")
    if limit is not None and limit < len(xml_paths):
        xml_paths = sorted(random.Random(seed).sample(xml_paths, limit))

    articles: dict[str, ParsedArticle] = {}
    skipped = 0
    for path in xml_paths:
        try:
            article = parse_article(path.read_bytes())
        except (ValueError, etree.XMLSyntaxError) as err:
            skipped += 1
            logger.warning("skipping %s: %s", path.name, err)
            continue
        articles[article.pmid] = article

    logger.info(
        "loaded corpus: %d articles (%d skipped) from %s",
        len(articles), skipped, raw_dir,
    )
    return articles


def load_chunk_index(
    raw_dir: Path, limit: int | None = None, seed: int = 42
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Rebuild the deterministic chunk index for a corpus subset.

    Returns (chunk_id -> content, pmid -> [chunk_id, ...]). Used by the Stage-2
    scripts to recover chunk text (for judging) and a paper's chunks (for pooling)
    without persisting the corpus — same limit/seed reproduces the same chunk_ids
    as `02_ingest.py`."""
    articles = load_corpus(raw_dir, limit=limit, seed=seed)
    chunks = chunk_all(articles)
    by_id = {c.chunk_id: c.content for c in chunks}
    by_pmid: dict[str, list[str]] = {}
    for chunk in chunks:
        by_pmid.setdefault(chunk.pmid, []).append(chunk.chunk_id)
    return by_id, by_pmid
