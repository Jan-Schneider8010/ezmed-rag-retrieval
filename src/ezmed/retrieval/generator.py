"""Answer generation from retrieved chunks (used for the qualitative sample)."""

from ezmed.llm.client import LLMClient
from ezmed.schemas import Chunk


class AnswerGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, query: str, chunks: list[Chunk]) -> tuple[str, int]:
        """Return (answer_text, tokens_used)."""
        raise NotImplementedError
