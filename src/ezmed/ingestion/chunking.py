"""Sliding-window chunking — 1000 chars / 200 overlap (configurable)."""

from collections.abc import Iterator

from ezmed.schemas import Chunk, Paper


def chunk_paper(paper: Paper, chunk_size: int, overlap: int) -> Iterator[Chunk]:
    """Yield section-aware sliding-window chunks for a paper."""
    raise NotImplementedError
