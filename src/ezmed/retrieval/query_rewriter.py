"""LLM-based query rewriting: lay phrasing → medical terminology."""

from ezmed.llm.client import LLMClient


class QueryRewriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite(self, lay_query: str) -> tuple[str, int]:
        """Return (rewritten_query, tokens_used)."""
        raise NotImplementedError
