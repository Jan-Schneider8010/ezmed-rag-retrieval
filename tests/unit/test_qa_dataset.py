from pathlib import Path
from unittest.mock import MagicMock

from ezmed.evaluation.qa_dataset import (
    MIN_ABSTRACT_CHARS,
    STRATEGIES,
    QADatasetBuilder,
    abstract_text,
    export_jsonl,
    load_jsonl,
)
from ezmed.llm.client import LLMResponse
from ezmed.schemas import ParsedArticle, Section


def _article(pmid: str, abstract_len: int = MIN_ABSTRACT_CHARS) -> ParsedArticle:
    return ParsedArticle(
        pmid=pmid,
        title=f"Paper {pmid}",
        abstract=[Section(title="Abstract", text="x" * abstract_len)] if abstract_len else [],
    )


def _corpus(n: int = 12) -> list[ParsedArticle]:
    return [_article(str(p)) for p in range(n)]


def test_abstract_text_joins_sections() -> None:
    article = ParsedArticle(
        pmid="1",
        title="t",
        abstract=[Section(title="A", text="one"), Section(title="B", text="two")],
    )
    assert abstract_text(article) == "one\ntwo"


def test_sampling_is_deterministic() -> None:
    builder = QADatasetBuilder(MagicMock(), n_questions=8, seed=7)
    a = builder.sample_papers(_corpus())
    b = builder.sample_papers(_corpus())
    assert [art.pmid for art, _ in a] == [art.pmid for art, _ in b]


def test_sampling_respects_count_and_strategy_round_robin() -> None:
    builder = QADatasetBuilder(MagicMock(), n_questions=9, seed=1)
    sampled = builder.sample_papers(_corpus())
    assert len(sampled) == 9
    assert [s for _, s in sampled] == [STRATEGIES[i % 3] for i in range(9)]


def test_sampling_filters_papers_without_usable_abstract() -> None:
    articles = [_article("1", abstract_len=MIN_ABSTRACT_CHARS - 1), _article("2")]
    builder = QADatasetBuilder(MagicMock(), n_questions=5)
    picked = {art.pmid for art, _ in builder.sample_papers(articles)}
    assert picked == {"2"}


def test_generate_sets_source_pmid_and_empty_gold() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        text='"What does aspirin do?"', prompt_tokens=1, completion_tokens=1
    )
    builder = QADatasetBuilder(llm, n_questions=3, seed=2)
    pairs = builder.generate(_corpus())
    assert len(pairs) == 3
    assert {p.qa_id for p in pairs} == {"qa_0000", "qa_0001", "qa_0002"}
    for pair in pairs:
        assert pair.relevant_chunk_ids == []
        assert pair.pmid in {str(p) for p in range(12)}
        assert pair.question == "What does aspirin do?"  # quotes stripped


def test_export_and_load_roundtrip(tmp_path: Path) -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text="q?", prompt_tokens=1, completion_tokens=1)
    pairs = QADatasetBuilder(llm, n_questions=4).generate(_corpus())
    path = export_jsonl(pairs, tmp_path / "qa.jsonl")
    loaded = load_jsonl(path)
    assert [p.qa_id for p in loaded] == [p.qa_id for p in pairs]
    assert [p.pmid for p in loaded] == [p.pmid for p in pairs]
    assert loaded[0].relevant_chunk_ids == []
