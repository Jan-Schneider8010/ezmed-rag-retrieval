"""Character-based sliding-window chunker."""

from collections.abc import Iterator

from ezmed.schemas import Chunk, ParsedArticle, Section


def chunk_article(
    article: ParsedArticle, chunk_size: int, overlap: int
) -> Iterator[Chunk]:
    """Yield chunks across all sections. Windows do not cross section boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    position = 0
    for section in _iter_sections(article):
        for char_start, char_end, content in _slide(section.text, chunk_size, overlap):
            yield Chunk(
                chunk_id=f"{article.pmid}:{position:04d}",
                pmid=article.pmid,
                section=section.title,
                position=position,
                char_start=char_start,
                char_end=char_end,
                content=content,
            )
            position += 1


def _iter_sections(article: ParsedArticle) -> Iterator[Section]:
    yield from article.abstract
    yield from article.body


def _slide(text: str, chunk_size: int, overlap: int) -> Iterator[tuple[int, int, str]]:
    if not text:
        return
    stride = chunk_size - overlap
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        yield start, end, text[start:end]
        if end == n:
            return
        start += stride
