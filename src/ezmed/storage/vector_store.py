"""Qdrant wrapper, one instance per collection."""

import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from ezmed.schemas import EmbeddedChunk

# Qdrant point IDs must be UUIDs or unsigned ints; we derive a deterministic
# UUID5 from each chunk_id and keep the original string in the payload.
_CHUNK_NAMESPACE = uuid.UUID("a3e8c1f4-7b25-4d3e-9c11-1a0fef93d8f0")


class VectorStore:
    def __init__(self, host: str, port: int, collection: str, dim: int) -> None:
        self.collection = collection
        self.dim = dim
        self._client = QdrantClient(host=host, port=port)

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self.collection):
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection)
        except (UnexpectedResponse, ValueError):
            pass
        self.ensure_collection()

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        points = [_to_point(c) for c in chunks]
        for batch in _batched(points, size=256):
            self._client.upsert(collection_name=self.collection, points=batch)

    def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        result = self._client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=top_k,
            with_payload=True,
        )
        return [(p.payload["chunk_id"], p.score) for p in result.points]


def _to_point(item: EmbeddedChunk) -> PointStruct:
    chunk = item.chunk
    point_id = str(uuid.uuid5(_CHUNK_NAMESPACE, chunk.chunk_id))
    return PointStruct(
        id=point_id,
        vector=item.embedding,
        payload={
            "chunk_id": chunk.chunk_id,
            "pmid": chunk.pmid,
            "section": chunk.section,
            "position": chunk.position,
            "content": chunk.content,
        },
    )


def _batched(items: list[PointStruct], size: int) -> Iterable[list[PointStruct]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
