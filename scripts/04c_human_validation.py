"""Step 4c: human validation of the judge-built gold.

--export: draw a blind, stratified sample of (question, chunk) pairs from the judge
labels (oversampling disagreements, the hardest cases) and write a CSV to fill in.
--score: read the filled CSV, compare your labels against the judge consensus, and
report Cohen's kappa + accuracy — the trust metric for the automated gold standard.

The exported CSV is blind: it does NOT contain the judge labels. Fill the empty
`human_relevant` column with 0 or 1, then run --score.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from random import Random

from ezmed.evaluation.analysis import cohen_kappa
from ezmed.evaluation.qa_dataset import load_jsonl
from ezmed.ingestion.corpus import load_chunk_index
from ezmed.logging import configure_logging

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
QA_PATH = Path("data/qa_dataset/qa_questions.jsonl")
LABELS_PATH = Path("results/ezmed_judge_labels.jsonl")
SAMPLE_PATH = Path("data/qa_dataset/validation_sample.csv")
FIELDS = ["qa_id", "chunk_id", "question", "chunk_text", "human_relevant"]


def main() -> None:
    args = _parse_args()
    configure_logging()
    if args.mode == "export":
        _export(args)
    else:
        _score(args)


def _export(args: argparse.Namespace) -> None:
    labels = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    sample = _stratified_sample(labels, args.n, args.seed)

    questions = {qa.qa_id: qa.question for qa in load_jsonl(args.qa)}
    chunk_text, _ = load_chunk_index(RAW_DIR, limit=args.corpus_limit, seed=args.seed)

    args.sample.parent.mkdir(parents=True, exist_ok=True)
    with args.sample.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sample:
            writer.writerow({
                "qa_id": row["qa_id"],
                "chunk_id": row["chunk_id"],
                "question": questions.get(row["qa_id"], ""),
                "chunk_text": chunk_text.get(row["chunk_id"], ""),
                "human_relevant": "",
            })
    logger.info(
        "wrote %d blind pairs to %s — fill 'human_relevant' (0/1), then run --score",
        len(sample), args.sample,
    )


def _stratified_sample(labels: list[dict], n: int, seed: int) -> list[dict]:
    rng = Random(seed)
    disagree = [x for x in labels if x["judge_a"] != x["judge_b"]]
    agree_rel = [x for x in labels if x["judge_a"] == x["judge_b"] and x["final"]]
    agree_not = [x for x in labels if x["judge_a"] == x["judge_b"] and not x["final"]]
    for bucket in (disagree, agree_rel, agree_not):
        rng.shuffle(bucket)

    picked = disagree[: n // 2]
    remaining = n - len(picked)
    picked += agree_rel[: remaining - remaining // 2]
    picked += agree_not[: remaining // 2]
    rng.shuffle(picked)
    return picked


def _score(args: argparse.Namespace) -> None:
    consensus = {
        (x["qa_id"], x["chunk_id"]): bool(x["final"])
        for x in (json.loads(line) for line in args.labels.read_text().splitlines() if line.strip())
    }
    human: list[bool] = []
    judge: list[bool] = []
    with args.sample.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            value = (row.get("human_relevant") or "").strip()
            if value not in {"0", "1"}:
                continue
            key = (row["qa_id"], row["chunk_id"])
            if key not in consensus:
                continue
            human.append(value == "1")
            judge.append(consensus[key])

    if not human:
        raise RuntimeError("no filled rows found in the sample CSV")

    kappa = cohen_kappa(human, judge)
    accuracy = sum(1 for h, j in zip(human, judge, strict=True) if h == j) / len(human)
    report = {
        "n_validated": len(human),
        "human_vs_judge_kappa": round(kappa, 4),
        "accuracy": round(accuracy, 4),
        "n_human_relevant": sum(human),
    }
    (RESULTS_DIR / "ezmed_human_validation.json").write_text(json.dumps(report, indent=2))
    logger.info(
        "validated %d pairs: human-vs-judge kappa=%.3f, accuracy=%.1f%%",
        len(human), kappa, accuracy * 100,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", dest="mode", action="store_const", const="export")
    group.add_argument("--score", dest="mode", action="store_const", const="score")
    parser.add_argument("--qa", type=Path, default=QA_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--n", type=int, default=80, help="Sample size for --export. Default 80.")
    parser.add_argument("--corpus-limit", type=int, default=100, help="Match step 2. Default 100.")
    parser.add_argument("--seed", type=int, default=42, help="Match step 2. Default 42.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
