"""Parse PMC JATS XML to a structured ParsedArticle (no filtering).

Filtering of sections (e.g. dropping introductions per §4.1 of the exposé)
belongs in the chunker, not here. This module is a pure transformation:
XML in, structured data out."""

from lxml import etree

from ezmed.ingestion.pubmed import (
    JATS_AUTHORS,
    JATS_DOI,
    JATS_JOURNAL,
    JATS_PMCID,
    JATS_PMID,
    JATS_TITLE,
    _format_author,
    _text_or_none,
)
from ezmed.schemas import ParsedArticle, Section

_STRIP_TAGS = ("table-wrap", "fig")


def parse_article(raw_xml: str | bytes) -> ParsedArticle:
    """Parse one JATS article into a ParsedArticle. Raises on malformed input."""
    if isinstance(raw_xml, str):
        raw_xml = raw_xml.encode("utf-8")
    root = etree.fromstring(raw_xml)
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        raise ValueError("no <article> element found")

    pmid_el = article.find(JATS_PMID)
    if pmid_el is None or not pmid_el.text:
        raise ValueError("article missing pmid")

    return ParsedArticle(
        pmid=pmid_el.text.strip(),
        pmcid=_optional_id(article, JATS_PMCID),
        doi=_optional_id(article, JATS_DOI),
        title=_text_or_none(article.find(JATS_TITLE)) or "",
        journal=_text_or_none(article.find(JATS_JOURNAL)),
        authors=[
            a
            for a in (_format_author(n) for n in article.findall(JATS_AUTHORS))
            if a
        ],
        abstract=_parse_abstract(article),
        body=_parse_body(article),
    )


def _parse_abstract(article: etree._Element) -> list[Section]:
    abstract = article.find(".//front//abstract")
    if abstract is None:
        return []
    etree.strip_elements(abstract, *_STRIP_TAGS, with_tail=False)
    secs = abstract.findall("sec")
    if secs:
        return [s for s in (_section_from_elem(sec) for sec in secs) if s]
    text = "".join(abstract.itertext()).strip()
    return [Section(title="Abstract", text=text)] if text else []


def _parse_body(article: etree._Element) -> list[Section]:
    body = article.find(".//body")
    if body is None:
        return []
    etree.strip_elements(body, *_STRIP_TAGS, with_tail=False)
    return [s for s in (_section_from_elem(sec) for sec in body.findall("sec")) if s]


def _section_from_elem(sec: etree._Element) -> Section | None:
    title_el = sec.find("title")
    if title_el is None:
        return None
    title = "".join(title_el.itertext()).strip()
    if not title:
        return None
    text = "".join(sec.itertext()).strip()
    if text.startswith(title):
        text = text[len(title) :].lstrip()
    if not text:
        return None
    return Section(title=title, text=text)


def _optional_id(article: etree._Element, xpath: str) -> str | None:
    el = article.find(xpath)
    if el is None or not el.text:
        return None
    return el.text.strip().removeprefix("PMC")
