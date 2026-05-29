"""Step 1: download ~1000 open-access PubMed papers for the chosen domain.

Reads PUBMED_DOMAIN and PUBMED_CORPUS_SIZE from .env. Writes raw JATS XML to
data/raw/{pmcid}.xml and the PMC-ID list to data/raw/_pmcids.txt."""

import logging
from pathlib import Path

from ezmed.settings import settings
from ezmed.ingestion.pubmed import RAW_XML_DIR, PubMedFetcher, _build_query

logger = logging.getLogger(__name__)


def main() -> None:
    query = _build_query(settings.pubmed_domain)
    print(f"query: {query}")
    print(f"target size: {settings.pubmed_corpus_size}")

    with PubMedFetcher(
        email=settings.pubmed_email,
        api_key=settings.pubmed_api_key or None,
    ) as fetcher:
        pmc_ids = fetcher.search(query, max_results=settings.pubmed_corpus_size)
        print(f"search returned {len(pmc_ids)} PMC IDs")

        papers = list(fetcher.fetch(pmc_ids))

    pmcids_path: Path = RAW_XML_DIR / "_pmcids.txt"
    pmcids_path.write_text("\n".join(pmc_ids) + "\n")

    print(f"papers parsed from JATS: {len(papers)}")
    print(f"XMLs cached in: {RAW_XML_DIR}")
    print(f"PMC-ID list:    {pmcids_path}")


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    main()
