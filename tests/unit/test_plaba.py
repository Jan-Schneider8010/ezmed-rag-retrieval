import json
from pathlib import Path

from ezmed.ingestion.plaba import collapse_to_pmids, load_plaba, subsample


def _write(tmp_path: Path) -> Path:
    data = {
        "1": {
            "question": "q1",
            "question_type": "x",
            "111": {"Title": "T1", "abstract": {"1": "a", "2": "b"}},
            "222": {"Title": "T2", "abstract": {"1": "c"}},
        },
        "2": {
            "question": "q2",
            "333": {"Title": "T3", "abstract": {"1": "d"}},
        },
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    return path


def test_load_plaba_maps_questions_and_gold(tmp_path: Path) -> None:
    articles, queries = load_plaba(_write(tmp_path))
    assert set(articles) == {"111", "222", "333"}
    assert articles["111"].abstract[0].text == "a b"
    assert queries[0] == {"qid": "1", "question": "q1", "gold_pmids": ["111", "222"]}
    assert queries[1]["gold_pmids"] == ["333"]


def test_collapse_to_pmids_dedupes_preserving_order() -> None:
    chunk_ids = ["111:0000", "111:0001", "222:0000", "111:0002"]
    assert collapse_to_pmids(chunk_ids) == ["111", "222"]


def test_subsample_keeps_first_n_and_their_abstracts(tmp_path: Path) -> None:
    articles, queries = load_plaba(_write(tmp_path))
    queries, articles = subsample(queries, articles, limit=1)
    assert [q["qid"] for q in queries] == ["1"]
    assert set(articles) == {"111", "222"}
