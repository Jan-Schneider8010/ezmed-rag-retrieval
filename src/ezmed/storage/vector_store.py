"""Qdrant wrapper for chunk vectors."""

from ezmed.schemas import EmbeddedChunk


class VectorStore:
    def __init__(self, host: str, port: int, collection: str, dim: int) -> None:
        self.host = host
        self.port = port
        self.collection = collection
        self.dim = dim

    def ensure_collection(self) -> None:
        raise NotImplementedError

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        raise NotImplementedError

    def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        raise NotImplementedError
