"""Hypothetical-question enrichment (HyPE-style).

For each chunk an LLM generates k hypothetical lay questions the chunk could
answer. Questions are embedded alongside the chunk content so query embeddings
align better with the ingested representation."""

from ezmed.llm.client import LLMClient
from ezmed.schemas import Chunk


class HypotheticalQuestionGenerator:
    def __init__(self, llm: LLMClient, k: int) -> None:
        self.llm = llm
        self.k = k

    def enrich(self, chunk: Chunk) -> Chunk:
        """Return a copy of the chunk with `hq_questions` populated."""
        raise NotImplementedError
