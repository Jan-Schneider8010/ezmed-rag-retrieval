"""PubMed/PMC OA fetcher (E-Utilities)."""

import json
import logging
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import httpx
from lxml import etree
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ezmed.schemas import Paper

NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EFETCH_BATCH_SIZE = 200
RAW_XML_DIR = Path("data/raw")
RATE_LIMIT_NO_KEY = 3
RATE_LIMIT_WITH_KEY = 10

JATS_PMID = ".//article-id[@pub-id-type='pmid']"
JATS_PMCID = ".//article-id[@pub-id-type='pmcid']"
JATS_DOI = ".//article-id[@pub-id-type='doi']"
JATS_TITLE = ".//front//title-group/article-title"
JATS_JOURNAL = ".//front//journal-meta//journal-title"
JATS_AUTHORS = ".//front//contrib-group//contrib[@contrib-type='author']/name"

MEDLINE_ARTICLE = ".//PubmedArticle"
MEDLINE_PMID = ".//MedlineCitation/PMID"
MEDLINE_MESH = ".//MeshHeading/DescriptorName"
MEDLINE_PUBDATE = ".//Article/Journal/JournalIssue/PubDate"

_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

logger = logging.getLogger(__name__)


def _build_query(domain: str) -> str:
    return f"{domain}[MeSH] AND open access[filter] AND english[lang]"


def _parse_month(text: str) -> int:
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    if text in _MONTH_NAMES:
        return _MONTH_NAMES[text]
    raise ValueError(f"unparseable month: {text!r}")


def _parse_pubdate(pubdate_elem: etree._Element | None) -> date | None:
    if pubdate_elem is None:
        return None

    medline = pubdate_elem.find("MedlineDate")
    if medline is not None and medline.text:
        first_token = medline.text.strip().split()[0]
        try:
            return date(int(first_token), 1, 1)
        except ValueError:
            logger.warning("unparseable MedlineDate: %r", medline.text)
            return None

    year_el = pubdate_elem.find("Year")
    if year_el is None or year_el.text is None:
        logger.warning("PubDate has neither Year nor MedlineDate")
        return None

    try:
        year = int(year_el.text)
    except ValueError:
        logger.warning("unparseable Year: %r", year_el.text)
        return None

    month = 1
    month_el = pubdate_elem.find("Month")
    if month_el is not None and month_el.text:
        try:
            month = _parse_month(month_el.text)
        except ValueError:
            logger.warning("unparseable Month: %r — defaulting to 1", month_el.text)

    day = 1
    day_el = pubdate_elem.find("Day")
    if day_el is not None and day_el.text:
        try:
            day = int(day_el.text)
        except ValueError:
            logger.warning("unparseable Day: %r — defaulting to 1", day_el.text)

    try:
        return date(year, month, day)
    except ValueError:
        logger.warning("invalid date components: y=%s m=%s d=%s", year, month, day)
        return date(year, 1, 1)


def _format_author(name_elem: etree._Element) -> str:
    sn = name_elem.find("surname")
    gn = name_elem.find("given-names")
    surname = (sn.text or "").strip() if sn is not None else ""
    given = (gn.text or "").strip() if gn is not None else ""
    if surname and given:
        return f"{surname}, {given}"
    return surname or given


def _text_or_none(elem: etree._Element | None) -> str | None:
    if elem is None:
        return None
    text = "".join(elem.itertext()).strip()
    return text or None


def _paper_from_jats(article: etree._Element) -> Paper | None:
    pmid_el = article.find(JATS_PMID)
    if pmid_el is None or not pmid_el.text:
        logger.warning("JATS article missing PMID — skipping")
        return None
    pmid = pmid_el.text.strip()

    title = _text_or_none(article.find(JATS_TITLE))
    if title is None:
        logger.warning("JATS article %s missing title — skipping", pmid)
        return None

    doi_el = article.find(JATS_DOI)
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

    journal = _text_or_none(article.find(JATS_JOURNAL))

    authors = [_format_author(n) for n in article.findall(JATS_AUTHORS)]
    authors = [a for a in authors if a]

    return Paper(
        pmid=pmid,
        doi=doi,
        title=title,
        journal=journal,
        authors=authors,
        mesh_terms=[],
        published_at=None,
        full_text=None,
    )


def _apply_medline_metadata(
    papers_by_pmid: dict[str, Paper], medline_xml: bytes
) -> None:
    root = etree.fromstring(medline_xml)
    for art in root.iterfind(MEDLINE_ARTICLE):
        pmid_el = art.find(MEDLINE_PMID)
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text.strip()
        paper = papers_by_pmid.get(pmid)
        if paper is None:
            logger.warning("MEDLINE returned PMID %s not in request set", pmid)
            continue
        paper.mesh_terms = [
            d.text.strip()
            for d in art.iterfind(MEDLINE_MESH)
            if d.text and d.text.strip()
        ]
        paper.published_at = _parse_pubdate(art.find(MEDLINE_PUBDATE))


class PubMedFetcher:
    """Fetch open-access PubMed papers for a given MeSH-defined subdomain."""

    def __init__(self, email: str, api_key: str | None = None) -> None:
        self._email = email
        self._api_key = api_key or None
        self._client = httpx.Client(base_url=NCBI_EUTILS_BASE, timeout=60.0)
        rate = RATE_LIMIT_WITH_KEY if self._api_key else RATE_LIMIT_NO_KEY
        self._min_interval = 1.0 / rate
        self._last_call_ts = 0.0

    def __enter__(self) -> "PubMedFetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    def _common_params(self) -> dict[str, str]:
        params = {"tool": "ezmed", "email": self._email}
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.HTTPStatusError)
        ),
        reraise=True,
    )
    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
    ) -> bytes:
        self._throttle()
        merged_params = self._common_params()
        if params:
            merged_params.update(params)
        if data is not None:
            merged_data = self._common_params()
            merged_data.update(data)
            response = self._client.request(method, endpoint, data=merged_data)
        else:
            response = self._client.request(method, endpoint, params=merged_params)
        response.raise_for_status()
        return response.content

    def search(self, query: str, max_results: int) -> list[str]:
        """ESearch on db=pmc; returns PMC-IDs (digit-only, no 'PMC' prefix)."""
        content = self._request(
            "GET",
            "/esearch.fcgi",
            params={
                "db": "pmc",
                "term": query,
                "retmax": str(max_results),
                "retmode": "json",
            },
        )
        payload = json.loads(content)
        return list(payload["esearchresult"]["idlist"])

    def fetch(self, pmc_ids: list[str]) -> Iterator[Paper]:
        """Yield Paper objects for the given PMC-IDs.

        Caches raw JATS XML at data/raw/{pmcid}.xml. full_text=None
        (parser.py's responsibility). MEDLINE metadata is fetched fresh
        each call.
        """
        if not pmc_ids:
            return

        RAW_XML_DIR.mkdir(parents=True, exist_ok=True)

        cached: list[tuple[str, etree._Element]] = []
        to_fetch: list[str] = []
        for pmcid in pmc_ids:
            cache_path = RAW_XML_DIR / f"{pmcid}.xml"
            if cache_path.exists():
                article = etree.parse(str(cache_path)).getroot()
                cached.append((pmcid, article))
            else:
                to_fetch.append(pmcid)

        all_papers: list[Paper] = []

        for pmcid, article in cached:
            paper = _paper_from_jats(article)
            if paper is not None:
                all_papers.append(paper)

        for batch_start in range(0, len(to_fetch), EFETCH_BATCH_SIZE):
            batch = to_fetch[batch_start : batch_start + EFETCH_BATCH_SIZE]
            jats_bundle = self._request(
                "POST",
                "/efetch.fcgi",
                data={"db": "pmc", "id": ",".join(batch), "retmode": "xml"},
            )
            for pmcid, _, article in self._split_jats_articles(jats_bundle):
                cache_path = RAW_XML_DIR / f"{pmcid}.xml"
                cache_path.write_bytes(
                    etree.tostring(
                        article, xml_declaration=True, encoding="utf-8"
                    )
                )
                paper = _paper_from_jats(article)
                if paper is not None:
                    all_papers.append(paper)

        if not all_papers:
            return

        papers_by_pmid = {p.pmid: p for p in all_papers}
        pmids = list(papers_by_pmid.keys())
        for batch_start in range(0, len(pmids), EFETCH_BATCH_SIZE):
            batch = pmids[batch_start : batch_start + EFETCH_BATCH_SIZE]
            medline_xml = self._request(
                "POST",
                "/efetch.fcgi",
                data={"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
            )
            _apply_medline_metadata(papers_by_pmid, medline_xml)

        yield from all_papers

    @staticmethod
    def _split_jats_articles(
        jats_xml_bundle: bytes,
    ) -> Iterator[tuple[str, str, etree._Element]]:
        root = etree.fromstring(jats_xml_bundle)
        for article in root.iterfind(".//article"):
            pmcid_el = article.find(JATS_PMCID)
            pmid_el = article.find(JATS_PMID)
            if pmcid_el is None or not pmcid_el.text:
                logger.warning("JATS article missing PMCID — skipping")
                continue
            pmcid = pmcid_el.text.strip().removeprefix("PMC")
            pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
            yield pmcid, pmid, article
