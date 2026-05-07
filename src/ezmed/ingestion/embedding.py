"""Embedding via OpenAI text-embedding-3-large (3072-dim)."""

from ezmed.schemas import Chunk, EmbeddedChunk


class Embedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Concatenate chunk content with its HQ questions before embedding."""
        raise NotImplementedError

    def embed_batch(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        raise NotImplementedError
