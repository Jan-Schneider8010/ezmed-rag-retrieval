"""LLM-based query rewriting: lay phrasing → medical terminology."""

from ezmed.llm.client import LLMClient
from ezmed.llm.prompts import load_prompt


class QueryRewriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite(self, lay_query: str) -> tuple[str, int]:
        """Return (rewritten_query, tokens_used)."""
        prompt = load_prompt("query_rewriting")
        response = self.llm.complete(
            prompt.system, prompt.render_user(lay_query=lay_query)
        )
        return response.text.strip(), response.total_tokens
