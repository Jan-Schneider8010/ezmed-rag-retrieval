import pytest

from ezmed.evaluation.analysis import cohen_kappa
from ezmed.evaluation.judge import (
    PoolItem,
    RelevanceJudge,
    _parse_label,
    build_pool,
    run_judging,
)
from ezmed.llm.client import LLMResponse
from ezmed.schemas import QAPair


def _qa(qa_id: str, pmid: str = "1") -> QAPair:
    return QAPair(
        qa_id=qa_id,
        pmid=pmid,
        question=f"q {qa_id}",
        prompting_strategy="naive_lay",
        relevant_chunk_ids=[],
    )


class _StubJudge:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def label(self, item: PoolItem) -> bool:
        return self.fn(item)


class _FakeClient:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        text = '{"relevant": 1}' if "REL" in user else '{"relevant": 0}'
        return LLMResponse(text=text, prompt_tokens=1, completion_tokens=1)


def test_parse_label_json_true() -> None:
    assert _parse_label('{"relevant": 1, "reason": "x"}') is True


def test_parse_label_json_false() -> None:
    assert _parse_label('{"relevant": 0}') is False


def test_parse_label_prose_fallback() -> None:
    assert _parse_label("relevant: 1") is True


def test_parse_label_defaults_false() -> None:
    assert _parse_label("cannot tell") is False


def test_build_pool_source_first_then_sorted_union() -> None:
    pool = build_pool(
        _qa("qa_0000"),
        retrieved={"baseline": ["c3", "c9"], "both": ["c9", "c1"]},
        source_chunk_ids=["c5", "c3"],
        k=2,
    )
    assert pool[:2] == ["c5", "c3"]  # source order preserved
    assert pool[2:] == ["c1", "c9"]  # union minus source, sorted; c3 deduped


def test_build_pool_respects_k() -> None:
    pool = build_pool(_qa("qa_0000"), {"baseline": ["a", "b", "c"]}, [], k=2)
    assert set(pool) == {"a", "b"}


def test_build_pool_deduplicates_across_and_within_variants() -> None:
    pool = build_pool(
        _qa("qa_0000"),
        retrieved={"baseline": ["c1", "c1", "c2"], "hq_only": ["c2", "c3"]},
        source_chunk_ids=["c3", "c3"],  # source-paper chunk also retrieved
        k=10,
    )
    assert pool == ["c3", "c1", "c2"]  # source first (deduped), then sorted union
    assert [pool.count(c) for c in ("c1", "c2", "c3")] == [1, 1, 1]


def test_build_pool_default_depth_is_ten() -> None:
    from ezmed.evaluation.judge import POOL_DEPTH

    assert POOL_DEPTH == 10
    ids = [f"c{i}" for i in range(15)]
    pool = build_pool(_qa("qa_0000"), {"baseline": ids}, [])
    assert len(pool) == 10  # default k = POOL_DEPTH


def test_relevance_judge_labels_via_client() -> None:
    judge = RelevanceJudge(_FakeClient(), "fake")
    assert judge.label(PoolItem("qa_0000", "c1", "q", "this is REL")) is True
    assert judge.label(PoolItem("qa_0000", "c2", "q", "nope")) is False


def test_run_judging_full_agreement_builds_gold() -> None:
    qas = [_qa("qa_0000"), _qa("qa_0001")]
    pools = {"qa_0000": ["c1", "c2"], "qa_0001": ["c3"]}
    chunk_text = {"c1": "a", "c2": "b", "c3": "c"}
    relevant = {"c1", "c3"}
    judge = lambda it: it.chunk_id in relevant  # noqa: E731
    a = _StubJudge("A", judge)
    b = _StubJudge("B", judge)
    tb = _StubJudge("TB", lambda it: True)

    updated, report = run_judging(qas, pools, chunk_text, a, b, tb, workers=1)

    assert report.n_pairs == 3
    assert report.n_disagree == 0
    assert report.inter_judge_kappa == 1.0
    gold = {qa.qa_id: qa.relevant_chunk_ids for qa in updated}
    assert gold == {"qa_0000": ["c1"], "qa_0001": ["c3"]}


def test_run_judging_tiebreak_resolves_disagreements() -> None:
    qas = [_qa("qa_0000")]
    pools = {"qa_0000": ["c1", "c2"]}
    chunk_text = {"c1": "a", "c2": "b"}
    a = _StubJudge("A", lambda it: True)
    b = _StubJudge("B", lambda it: False)
    tb = _StubJudge("TB", lambda it: it.chunk_id == "c1")

    updated, report = run_judging(qas, pools, chunk_text, a, b, tb, workers=1)

    assert report.n_disagree == 2
    assert updated[0].relevant_chunk_ids == ["c1"]


def test_cohen_kappa_perfect() -> None:
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0


def test_cohen_kappa_all_same_label_is_one() -> None:
    assert cohen_kappa([True, True], [True, True]) == 1.0


def test_cohen_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])
