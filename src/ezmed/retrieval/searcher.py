"""Dense vector search."""

from ezmed.llm.client import LLMClient
from ezmed.storage.vector_store import VectorStore


class Searcher:
    def __init__(self, embedder: LLMClient, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        [embedding] = self.embedder.embed([query])
        return self.store.search(embedding, top_k)
