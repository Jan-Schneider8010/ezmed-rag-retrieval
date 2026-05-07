"""Ingestion orchestrator: papers → chunks → (HQ) → embeddings → Qdrant + Postgres."""

from collections.abc import Iterable

from ezmed.schemas import Paper, Variant


class IngestionPipeline:
    def __init__(self, variant: Variant) -> None:
        self.variant = variant

    def run(self, papers: Iterable[Paper]) -> None:
        raise NotImplementedError
