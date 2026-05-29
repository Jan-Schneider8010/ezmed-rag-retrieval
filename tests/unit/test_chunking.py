import pytest

from ezmed.ingestion.chunking import chunk_article
from ezmed.schemas import ParsedArticle, Section


def _article(*sections: tuple[str, str], pmid: str = "PMID1") -> ParsedArticle:
    body = [Section(title=title, text=text) for title, text in sections]
    return ParsedArticle(pmid=pmid, title="t", body=body)


def test_short_section_yields_single_chunk() -> None:
    article = _article(("Intro", "short text"))
    chunks = list(chunk_article(article, chunk_size=1000, overlap=200))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.content == "short text"
    assert c.char_start == 0
    assert c.char_end == 10
    assert c.section == "Intro"
    assert c.position == 0
    assert c.chunk_id == "PMID1:0000"


def test_empty_section_yields_no_chunks() -> None:
    article = _article(("Intro", ""))
    assert list(chunk_article(article, chunk_size=1000, overlap=200)) == []


def test_sliding_respects_size_and_overlap() -> None:
    text = "x" * 2500
    article = _article(("S", text))
    chunks = list(chunk_article(article, chunk_size=1000, overlap=200))
    assert [(c.char_start, c.char_end) for c in chunks] == [
        (0, 1000),
        (800, 1800),
        (1600, 2500),
    ]
    for prev, curr in zip(chunks, chunks[1:]):
        if curr.char_end < len(text) or curr.char_end - curr.char_start == 1000:
            assert prev.char_end - curr.char_start == 200


def test_section_boundary_is_respected() -> None:
    article = _article(("A", "x" * 1200), ("B", "y" * 500))
    chunks = list(chunk_article(article, chunk_size=1000, overlap=200))
    assert [c.section for c in chunks] == ["A", "A", "B"]
    a_chunks = [c for c in chunks if c.section == "A"]
    b_chunks = [c for c in chunks if c.section == "B"]
    assert all(c.content.startswith("x") for c in a_chunks)
    assert all(c.content.startswith("y") for c in b_chunks)


def test_position_is_globally_monotonic_across_sections() -> None:
    article = _article(("A", "x" * 1200), ("B", "y" * 500))
    chunks = list(chunk_article(article, chunk_size=1000, overlap=200))
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [
        "PMID1:0000",
        "PMID1:0001",
        "PMID1:0002",
    ]


def test_iterates_abstract_before_body() -> None:
    article = ParsedArticle(
        pmid="P",
        title="t",
        abstract=[Section(title="Abstract", text="a")],
        body=[Section(title="Body", text="b")],
    )
    chunks = list(chunk_article(article, chunk_size=10, overlap=2))
    assert [c.section for c in chunks] == ["Abstract", "Body"]


def test_rejects_invalid_overlap() -> None:
    article = _article(("S", "x"))
    with pytest.raises(ValueError):
        list(chunk_article(article, chunk_size=100, overlap=100))
    with pytest.raises(ValueError):
        list(chunk_article(article, chunk_size=100, overlap=-1))


def test_rejects_non_positive_chunk_size() -> None:
    article = _article(("S", "x"))
    with pytest.raises(ValueError):
        list(chunk_article(article, chunk_size=0, overlap=0))
