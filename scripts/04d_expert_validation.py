"""Step 4d: external validation of the judge-built gold by a medical professional.

Same idea as 04c (blind sample -> Cohen's kappa vs. the judge consensus) but built
to be handed to a domain expert instead of self-validated:

--export draws a blind, stratified sample of (question, chunk) pairs, oversampling
the judge *disagreements* (the cases the o3 tie-break decided — the ones an expert
most usefully checks) and spreading pairs across questions. It writes a German,
Excel-friendly CSV (UTF-8 BOM) whose only column to fill is `relevant` (0/1). The
file is blind: it does NOT contain any judge label.

--score reads the returned CSV, aligns each row to the judge consensus by
(qa_id, chunk_id), and reports Cohen's kappa + accuracy overall AND split by
stratum — crucially expert-vs-tiebreak on the disagreement pairs. Written to
results/ezmed_expert_validation.json. This is the external trust metric for the
Stage-2 gold standard (protocol §5, and the §4 fallback for a low inter-judge kappa).
"""

import argparse
import csv
import json
import logging
from collections import Counter
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
SAMPLE_PATH = Path("data/qa_dataset/expert_validation_sample.csv")

# German, expert-facing columns. Only `relevant` (and optionally `kommentar`) is filled.
FIELDS = ["nr", "qa_id", "chunk_id", "frage", "textausschnitt", "relevant", "kommentar"]


def main() -> None:
    args = _parse_args()
    configure_logging()
    if args.mode == "export":
        _export(args)
    else:
        _score(args)


def _export(args: argparse.Namespace) -> None:
    labels = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    sample = _stratified_sample(labels, args.n, args.seed, args.max_per_qa)

    questions = {qa.qa_id: qa.question for qa in load_jsonl(args.qa)}
    chunk_text, _ = load_chunk_index(RAW_DIR, limit=args.corpus_limit, seed=args.seed)

    args.sample.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel renders umlauts; blind = no judge columns.
    with args.sample.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(sample, start=1):
            writer.writerow({
                "nr": i,
                "qa_id": row["qa_id"],
                "chunk_id": row["chunk_id"],
                "frage": questions.get(row["qa_id"], ""),
                "textausschnitt": chunk_text.get(row["chunk_id"], ""),
                "relevant": "",
                "kommentar": "",
            })

    strata = Counter(_stratum(x) for x in sample)
    logger.info(
        "wrote %d blind pairs to %s (disagree=%d, relevant=%d, not_relevant=%d) "
        "across %d questions — expert fills 'relevant' (0/1), then run --score",
        len(sample), args.sample,
        strata["disagree"], strata["agree_rel"], strata["agree_not"],
        len({x["qa_id"] for x in sample}),
    )


def _stratum(x: dict) -> str:
    if x["judge_a"] != x["judge_b"]:
        return "disagree"
    return "agree_rel" if x["final"] else "agree_not"


def _stratified_sample(labels: list[dict], n: int, seed: int, max_per_qa: int) -> list[dict]:
    """Half disagreements (the tie-break cases), the rest split relevant/not-relevant.

    Oversamples disagreements because those are what an expert most usefully checks,
    and caps pairs-per-question so the sample spreads across the question set rather
    than clustering on a few long papers."""
    rng = Random(seed)
    buckets = {
        "disagree": [x for x in labels if x["judge_a"] != x["judge_b"]],
        "agree_rel": [x for x in labels if x["judge_a"] == x["judge_b"] and x["final"]],
        "agree_not": [x for x in labels if x["judge_a"] == x["judge_b"] and not x["final"]],
    }
    for bucket in buckets.values():
        rng.shuffle(bucket)

    remaining = n - n // 2
    targets = {"disagree": n // 2, "agree_rel": remaining - remaining // 2, "agree_not": remaining // 2}

    picked: list[dict] = []
    per_qa: Counter = Counter()
    for name, target in targets.items():
        taken = 0
        for x in buckets[name]:
            if taken >= target:
                break
            if per_qa[x["qa_id"]] >= max_per_qa:
                continue
            picked.append(x)
            per_qa[x["qa_id"]] += 1
            taken += 1
    rng.shuffle(picked)
    return picked


def _score(args: argparse.Namespace) -> None:
    labels = {
        (x["qa_id"], x["chunk_id"]): x
        for x in (json.loads(line) for line in args.labels.read_text().splitlines() if line.strip())
    }

    rows: list[tuple[bool, dict]] = []
    with args.sample.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            value = (row.get("relevant") or "").strip()
            if value not in {"0", "1"}:
                continue
            key = (row["qa_id"], row["chunk_id"])
            if key not in labels:
                continue
            rows.append((value == "1", labels[key]))

    if not rows:
        raise RuntimeError("no filled rows found in the sample CSV (fill the 'relevant' column with 0/1)")

    report = {
        "n_validated": len(rows),
        "n_expert_relevant": sum(1 for h, _ in rows if h),
        "overall": _pair_stats([(h, bool(x["final"])) for h, x in rows]),
        "on_disagreements_vs_tiebreak": _pair_stats(
            [(h, bool(x["final"])) for h, x in rows if x["judge_a"] != x["judge_b"]]
        ),
        "on_agreements_vs_consensus": _pair_stats(
            [(h, bool(x["final"])) for h, x in rows if x["judge_a"] == x["judge_b"]]
        ),
        "by_stratum": {
            name: _pair_stats([(h, bool(x["final"])) for h, x in rows if _stratum(x) == name])
            for name in ("disagree", "agree_rel", "agree_not")
        },
        "error_direction": _error_direction(rows),
        # The sample oversamples disagreements by design, so `overall` describes the
        # sample, not the gold set. Reweighting each stratum's accuracy by its share of
        # the full label set is the only figure that generalises to the corpus.
        "population_weighted": _population_weighted(rows, list(labels.values())),
    }
    (RESULTS_DIR / "ezmed_expert_validation.json").write_text(json.dumps(report, indent=2))
    logger.info(
        "validated %d pairs: sample kappa=%.3f, accuracy=%.1f%% (tie-break cases: kappa=%.3f, n=%d); "
        "population-weighted accuracy=%.1f%%; mismatches gold-liberal=%d vs gold-strict=%d",
        report["n_validated"], report["overall"]["kappa"], report["overall"]["accuracy"] * 100,
        report["on_disagreements_vs_tiebreak"]["kappa"], report["on_disagreements_vs_tiebreak"]["n"],
        report["population_weighted"]["accuracy"] * 100,
        report["error_direction"]["gold_relevant_expert_not"],
        report["error_direction"]["gold_not_expert_relevant"],
    )


def _error_direction(rows: list[tuple[bool, dict]]) -> dict:
    """Split the mismatches by sign: is the judge gold too liberal or too strict?"""
    return {
        "both_relevant": sum(1 for h, x in rows if h and x["final"]),
        "both_not_relevant": sum(1 for h, x in rows if not h and not x["final"]),
        "gold_relevant_expert_not": sum(1 for h, x in rows if not h and x["final"]),
        "gold_not_expert_relevant": sum(1 for h, x in rows if h and not x["final"]),
        "expert_relevant_rate": round(sum(1 for h, _ in rows if h) / len(rows), 4),
        "gold_relevant_rate": round(sum(1 for _, x in rows if x["final"]) / len(rows), 4),
    }


def _population_weighted(rows: list[tuple[bool, dict]], all_labels: list[dict]) -> dict:
    """Reweight per-stratum accuracy by each stratum's share of the full label set."""
    population = Counter(_stratum(x) for x in all_labels)
    total = sum(population.values())
    accuracy, covered, strata = 0.0, 0.0, {}
    for name, count in population.items():
        subset = [(h, x) for h, x in rows if _stratum(x) == name]
        share = count / total
        strata[name] = {"share": round(share, 4), "n_sampled": len(subset)}
        if not subset:
            continue
        stratum_accuracy = sum(1 for h, x in subset if h == bool(x["final"])) / len(subset)
        strata[name]["accuracy"] = round(stratum_accuracy, 4)
        accuracy += share * stratum_accuracy
        covered += share
    return {
        "accuracy": round(accuracy / covered, 4) if covered else None,
        "population_size": total,
        "strata": strata,
    }


def _pair_stats(pairs: list[tuple[bool, bool]]) -> dict:
    if not pairs:
        return {"n": 0, "kappa": None, "accuracy": None}
    expert = [h for h, _ in pairs]
    gold = [g for _, g in pairs]
    accuracy = sum(1 for h, g in pairs if h == g) / len(pairs)
    return {"n": len(pairs), "kappa": round(cohen_kappa(expert, gold), 4), "accuracy": round(accuracy, 4)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", dest="mode", action="store_const", const="export")
    group.add_argument("--score", dest="mode", action="store_const", const="score")
    parser.add_argument("--qa", type=Path, default=QA_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--n", type=int, default=150, help="Sample size for --export. Default 150.")
    parser.add_argument("--max-per-qa", type=int, default=3, help="Cap pairs per question. Default 3.")
    parser.add_argument("--corpus-limit", type=int, default=500, help="Match the run being validated. Default 500 (large run).")
    parser.add_argument("--seed", type=int, default=42, help="Match the run. Default 42.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
