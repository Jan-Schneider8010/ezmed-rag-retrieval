from unittest.mock import MagicMock

from ezmed.ingestion.enrichment import HypotheticalQuestionGenerator, enriched_text
from ezmed.llm.client import LLMResponse
from ezmed.schemas import Chunk


def _chunk(content: str = "Aspirin reduces clotting.", hq: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id="1:0000",
        pmid="1",
        section="Abstract",
        position=0,
        char_start=0,
        char_end=len(content),
        content=content,
        hq_questions=hq or [],
    )


def test_enrich_parses_lines_strips_numbering_and_limits_to_k() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        text="1. What is aspirin?\n2. Does it help?\n\n3. extra\n4. four",
        prompt_tokens=1,
        completion_tokens=1,
    )
    out = HypotheticalQuestionGenerator(llm, k=3).enrich(_chunk())
    assert out.hq_questions == ["What is aspirin?", "Does it help?", "extra"]


def test_enrich_does_not_mutate_original_chunk() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text="q one", prompt_tokens=1, completion_tokens=1)
    original = _chunk()
    out = HypotheticalQuestionGenerator(llm, k=2).enrich(original)
    assert original.hq_questions == []
    assert out.hq_questions == ["q one"]


def test_enriched_text_concats_content_and_questions() -> None:
    chunk = _chunk(hq=["q1", "q2"])
    assert enriched_text(chunk) == "Aspirin reduces clotting.\nq1\nq2"


def test_enriched_text_without_questions_is_content() -> None:
    assert enriched_text(_chunk()) == "Aspirin reduces clotting."
