"""Tests for the JATS XML parser."""

from pathlib import Path

import pytest

from ezmed.ingestion.parser import parse_article
from ezmed.schemas import ParsedArticle


def _load(fixtures_dir: Path) -> bytes:
    return (fixtures_dir / "sample_pmc.xml").read_bytes()


def test_parse_metadata_extracted(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    assert article.pmid == "41891318"
    assert article.pmcid == "13140896"
    assert article.doi == "10.1002/cam4.71682"
    assert article.title.startswith("Standardizing")
    assert article.journal == "Cancer Medicine"


def test_parse_authors_non_empty(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    assert "Ali, Abdelrahman" in article.authors
    assert len(article.authors) >= 5


def test_parse_abstract_has_sections(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    titles = [s.title for s in article.abstract]
    assert titles == [
        "Introduction and Methods",
        "Clinical Scenarios and Discussion",
        "Conclusion",
    ]
    assert all(s.text for s in article.abstract)


def test_parse_abstract_flat_returns_single_section() -> None:
    xml = b"""<article>
      <front><article-meta>
        <article-id pub-id-type="pmid">1</article-id>
        <title-group><article-title>x</article-title></title-group>
        <abstract><p>One paragraph.</p><p>Two paragraph.</p></abstract>
      </article-meta></front>
    </article>"""
    article = parse_article(xml)
    assert len(article.abstract) == 1
    assert article.abstract[0].title == "Abstract"
    assert "One paragraph" in article.abstract[0].text
    assert "Two paragraph" in article.abstract[0].text


def test_parse_body_includes_introduction(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    titles = [s.title for s in article.body]
    assert "Introduction" in titles


def test_parse_body_includes_back_matter(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    titles = [s.title for s in article.body]
    assert "Funding" in titles
    assert "Author Contributions" in titles
    assert "Conflicts of Interest" in titles


def test_parse_body_strips_tables(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    combined = "\n".join(s.text for s in article.body)
    assert "TABLE 1" not in combined
    assert "TABLE 5" not in combined
    assert "Symptoms and signs of heart failure" not in combined


def test_parse_body_strips_figures(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    combined = "\n".join(s.text for s in article.body)
    assert "FIGURE 1" not in combined
    assert "Diagnostic algorithm for cancer therapy" not in combined


def test_section_text_does_not_start_with_title(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    for sec in article.body:
        assert not sec.text.startswith(sec.title), (
            f"section {sec.title!r} text starts with its own title"
        )


def test_parse_skips_section_without_title() -> None:
    xml = b"""<article>
      <front><article-meta>
        <article-id pub-id-type="pmid">1</article-id>
        <title-group><article-title>x</article-title></title-group>
      </article-meta></front>
      <body>
        <sec><p>orphan paragraph</p></sec>
        <sec><title>Real</title><p>kept</p></sec>
      </body>
    </article>"""
    article = parse_article(xml)
    assert [s.title for s in article.body] == ["Real"]


def test_parse_raises_when_no_article() -> None:
    with pytest.raises(ValueError, match="no <article>"):
        parse_article(b"<root><x/></root>")


def test_parse_raises_when_no_pmid() -> None:
    xml = b"""<article>
      <front><article-meta>
        <title-group><article-title>x</article-title></title-group>
      </article-meta></front>
    </article>"""
    with pytest.raises(ValueError, match="missing pmid"):
        parse_article(xml)


def test_parse_handles_str_and_bytes_input(fixtures_dir: Path) -> None:
    raw_bytes = _load(fixtures_dir)
    raw_str = raw_bytes.decode("utf-8")
    a_bytes = parse_article(raw_bytes)
    a_str = parse_article(raw_str)
    assert a_bytes == a_str


def test_parsed_article_round_trips_to_json(fixtures_dir: Path) -> None:
    article = parse_article(_load(fixtures_dir))
    payload = article.model_dump_json()
    restored = ParsedArticle.model_validate_json(payload)
    assert restored == article
