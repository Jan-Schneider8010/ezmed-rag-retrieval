"""Hypothetical-question enrichment (HyPE-style).

For each chunk an LLM generates k hypothetical lay questions the chunk could
answer. Questions are embedded alongside the chunk content so query embeddings
align better with the ingested representation."""

from ezmed.llm.client import LLMClient
from ezmed.llm.prompts import load_prompt
from ezmed.schemas import Chunk


class HypotheticalQuestionGenerator:
    def __init__(self, llm: LLMClient, k: int) -> None:
        self.llm = llm
        self.k = k

    def enrich(self, chunk: Chunk) -> Chunk:
        """Return a copy of the chunk with `hq_questions` populated."""
        prompt = load_prompt("hq_generation")
        response = self.llm.complete(
            prompt.system, prompt.render_user(chunk_text=chunk.content, k=self.k)
        )
        questions = _parse_questions(response.text, self.k)
        return chunk.model_copy(update={"hq_questions": questions})


def _parse_questions(text: str, k: int) -> list[str]:
    questions: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("0123456789.-) ").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:k]


def enriched_text(chunk: Chunk) -> str:
    """Concat chunk content with its HQ questions (Variant A: one vector per chunk)."""
    if not chunk.hq_questions:
        return chunk.content
    return chunk.content + "\n" + "\n".join(chunk.hq_questions)
