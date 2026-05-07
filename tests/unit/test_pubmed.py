"""Tests for the PubMed/PMC OA fetcher."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx
from lxml import etree

from ezmed.ingestion import pubmed
from ezmed.ingestion.pubmed import (
    PubMedFetcher,
    _apply_medline_metadata,
    _build_query,
    _format_author,
    _paper_from_jats,
    _parse_pubdate,
)
from ezmed.schemas import Paper

PUBDATE_CLEAN = b"""<PubDate><Year>2023</Year><Month>Jan</Month><Day>15</Day></PubDate>"""
PUBDATE_PARTIAL_YM = b"""<PubDate><Year>2024</Year><Month>3</Month></PubDate>"""
PUBDATE_PARTIAL_Y = b"""<PubDate><Year>2025</Year></PubDate>"""
PUBDATE_MEDLINE = b"""<PubDate><MedlineDate>2023 Jan-Feb</MedlineDate></PubDate>"""
PUBDATE_MEDLINE_BAD = b"""<PubDate><MedlineDate>not a date</MedlineDate></PubDate>"""


def _load(path: Path) -> bytes:
    return path.read_bytes()


def test_build_query() -> None:
    assert (
        _build_query("cardiology")
        == "cardiology[MeSH] AND open access[filter] AND english[lang]"
    )


def test_paper_from_jats_extracts_all_fields(fixtures_dir: Path) -> None:
    tree = etree.parse(str(fixtures_dir / "sample_pmc.xml"))
    article = tree.findall(".//article")[0]
    paper = _paper_from_jats(article)
    assert paper is not None
    assert paper.pmid == "41891318"
    assert paper.doi == "10.1002/cam4.71682"
    assert "Cancer Therapy" in paper.title
    assert paper.journal == "Cancer Medicine"
    assert "Ali, Abdelrahman" in paper.authors
    assert len(paper.authors) >= 1
    assert paper.mesh_terms == []
    assert paper.published_at is None
    assert paper.full_text is None


def test_format_author_full() -> None:
    elem = etree.fromstring(
        b"<name><surname>Doe</surname><given-names>John Q.</given-names></name>"
    )
    assert _format_author(elem) == "Doe, John Q."


def test_format_author_surname_only() -> None:
    elem = etree.fromstring(b"<name><surname>Doe</surname></name>")
    assert _format_author(elem) == "Doe"


def test_format_author_hyphenated_given() -> None:
    elem = etree.fromstring(
        b"<name><surname>Doe</surname><given-names>Jean-Luc</given-names></name>"
    )
    assert _format_author(elem) == "Doe, Jean-Luc"


def test_parse_pubdate_clean() -> None:
    elem = etree.fromstring(PUBDATE_CLEAN)
    assert _parse_pubdate(elem) == date(2023, 1, 15)


def test_parse_pubdate_partial_year_month() -> None:
    elem = etree.fromstring(PUBDATE_PARTIAL_YM)
    assert _parse_pubdate(elem) == date(2024, 3, 1)


def test_parse_pubdate_partial_year_only() -> None:
    elem = etree.fromstring(PUBDATE_PARTIAL_Y)
    assert _parse_pubdate(elem) == date(2025, 1, 1)


def test_parse_pubdate_medlinedate() -> None:
    elem = etree.fromstring(PUBDATE_MEDLINE)
    assert _parse_pubdate(elem) == date(2023, 1, 1)


def test_parse_pubdate_medlinedate_unparseable() -> None:
    elem = etree.fromstring(PUBDATE_MEDLINE_BAD)
    assert _parse_pubdate(elem) is None


def test_apply_medline_metadata(fixtures_dir: Path) -> None:
    medline_xml = _load(fixtures_dir / "sample_medline.xml")
    papers = {
        "41891318": Paper(pmid="41891318", title="placeholder"),
        "41578910": Paper(pmid="41578910", title="placeholder"),
    }
    _apply_medline_metadata(papers, medline_xml)
    p1 = papers["41891318"]
    assert len(p1.mesh_terms) > 0
    assert "Humans" in p1.mesh_terms
    assert p1.published_at == date(2026, 4, 1)
    p2 = papers["41578910"]
    assert len(p2.mesh_terms) > 0
    assert p2.published_at == date(2026, 5, 1)


@respx.mock
def test_search_returns_pmc_ids() -> None:
    payload = {
        "esearchresult": {
            "count": "2",
            "retmax": "2",
            "retstart": "0",
            "idlist": ["13140896", "13135285"],
        }
    }
    route = respx.get(f"{pubmed.NCBI_EUTILS_BASE}/esearch.fcgi").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with PubMedFetcher(email="jan@example.com", api_key="key123") as f:
        ids = f.search("cardiology[MeSH]", max_results=2)
    assert ids == ["13140896", "13135285"]
    assert route.call_count == 1
    request = route.calls.last.request
    params = dict(request.url.params)
    assert params["db"] == "pmc"
    assert params["term"] == "cardiology[MeSH]"
    assert params["retmode"] == "json"
    assert params["retmax"] == "2"
    assert params["tool"] == "ezmed"
    assert params["email"] == "jan@example.com"
    assert params["api_key"] == "key123"


@respx.mock
def test_fetch_writes_raw_xml_and_caches(
    tmp_path: Path,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pubmed, "RAW_XML_DIR", tmp_path)

    pmc_xml = _load(fixtures_dir / "sample_pmc.xml")
    medline_xml = _load(fixtures_dir / "sample_medline.xml")

    pmc_route = respx.post(f"{pubmed.NCBI_EUTILS_BASE}/efetch.fcgi").mock(
        side_effect=lambda req: (
            httpx.Response(200, content=pmc_xml)
            if b"db=pmc" in req.content
            else httpx.Response(200, content=medline_xml)
        )
    )

    with PubMedFetcher(email="jan@example.com") as f:
        papers = list(f.fetch(["13140896", "13135285"]))

    assert len(papers) == 2
    pmids = {p.pmid for p in papers}
    assert pmids == {"41891318", "41578910"}

    p1 = next(p for p in papers if p.pmid == "41891318")
    assert p1.title.startswith("Standardizing")
    assert p1.journal == "Cancer Medicine"
    assert len(p1.mesh_terms) > 0
    assert p1.published_at == date(2026, 4, 1)
    assert p1.full_text is None
    assert len(p1.authors) >= 1

    assert (tmp_path / "13140896.xml").exists()
    assert (tmp_path / "13135285.xml").exists()
    assert pmc_route.call_count == 2
    pmc_calls = sum(1 for c in pmc_route.calls if b"db=pmc" in c.request.content)
    medline_calls = sum(
        1 for c in pmc_route.calls if b"db=pubmed" in c.request.content
    )
    assert pmc_calls == 1
    assert medline_calls == 1

    with PubMedFetcher(email="jan@example.com") as f:
        papers2 = list(f.fetch(["13140896", "13135285"]))

    assert len(papers2) == 2
    pmc_calls_after = sum(
        1 for c in pmc_route.calls if b"db=pmc" in c.request.content
    )
    medline_calls_after = sum(
        1 for c in pmc_route.calls if b"db=pubmed" in c.request.content
    )
    assert pmc_calls_after == 1, "second run should hit PMC cache"
    assert medline_calls_after == 2, "MEDLINE is intentionally re-fetched"


def test_rate_limit_throttles(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    fake_now = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        fake_now[0] += s

    monkeypatch.setattr(pubmed.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(pubmed.time, "sleep", fake_sleep)

    f = PubMedFetcher(email="jan@example.com")
    try:
        f._throttle()
        f._throttle()
        f._throttle()
    finally:
        f.close()

    expected = 1.0 / pubmed.RATE_LIMIT_NO_KEY
    assert sleeps[0] >= 0
    assert sleeps[1] >= expected - 1e-6
    assert sleeps[2] >= expected - 1e-6


@respx.mock
def test_retry_on_5xx() -> None:
    payload = {"esearchresult": {"idlist": ["1"]}}
    route = respx.get(f"{pubmed.NCBI_EUTILS_BASE}/esearch.fcgi").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json=payload),
        ]
    )
    with PubMedFetcher(email="jan@example.com") as f:
        ids = f.search("foo", max_results=1)
    assert ids == ["1"]
    assert route.call_count == 3
