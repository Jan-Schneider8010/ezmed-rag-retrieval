"""Dense vector search against Qdrant."""

from ezmed.ingestion.embedding import Embedder
from ezmed.storage.vector_store import VectorStore


class Searcher:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, score), ...] ranked by similarity."""
        raise NotImplementedError
