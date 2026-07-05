"""Step 5: score the four variants against the judge-built gold and test significance.

Reads step-4 retrieval and the step-4b gold, computes Recall@5/10, MRR, NDCG@10 per
variant at chunk granularity, and the pairwise Wilcoxon/Bonferroni ablation. Writes
results/ezmed_<variant>.json and results/ezmed_ablation.json. Questions whose gold is
empty (no pooled chunk judged relevant) are excluded — retrieval quality is undefined
without gold.
"""

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

from ezmed.evaluation.analysis import build_ablation
from ezmed.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from ezmed.evaluation.qa_dataset import load_jsonl
from ezmed.logging import configure_logging

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RETRIEVAL_PATH = Path("results/ezmed_retrieval.json")
GOLD_PATH = Path("data/qa_dataset/qa_gold.jsonl")
VARIANTS = ("baseline", "qr_only", "hq_only", "both")


def main() -> None:
    args = _parse_args()
    configure_logging()

    retrieval: dict[str, dict[str, list[str]]] = json.loads(args.retrieval.read_text())
    gold = {qa.qa_id: set(qa.relevant_chunk_ids) for qa in load_jsonl(args.gold)}

    scored_ids = sorted(
        qa_id for qa_id, rel in gold.items() if rel and qa_id in retrieval
    )
    excluded = len(gold) - len(scored_ids)
    if not scored_ids:
        raise RuntimeError("no questions with non-empty gold to score")

    per_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        per_query = [_score_query(qa_id, retrieval[qa_id][variant], gold[qa_id]) for qa_id in scored_ids]
        per_variant[variant] = per_query
        metrics = _aggregate(per_query)
        _write_variant(variant, metrics, per_query, args.top_k)
        logger.info(
            "%s: Recall@5=%.3f Recall@10=%.3f MRR=%.3f NDCG@10=%.3f",
            variant, metrics["recall_at_5"], metrics["recall_at_10"],
            metrics["mrr"], metrics["ndcg_at_10"],
        )

    ablation = build_ablation(per_variant, id_key="qa_id", dataset="ezmed")
    ablation["n_excluded_empty_gold"] = excluded
    (RESULTS_DIR / "ezmed_ablation.json").write_text(json.dumps(ablation, indent=2))
    logger.info(
        "scored %d questions (%d excluded, empty gold); wrote ezmed_ablation.json",
        len(scored_ids), excluded,
    )


def _score_query(qa_id: str, retrieved: list[str], relevant: set[str]) -> dict[str, Any]:
    return {
        "qa_id": qa_id,
        "n_gold": len(relevant),
        "recall_at_5": recall_at_k(retrieved, relevant, 5),
        "recall_at_10": recall_at_k(retrieved, relevant, 10),
        "mrr": reciprocal_rank(retrieved, relevant),
        "ndcg_at_10": ndcg_at_k(retrieved, relevant, 10),
    }


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    return {k: mean(q[k] for q in per_query) for k in keys}


def _write_variant(
    variant: str, metrics: dict[str, float], per_query: list[dict[str, Any]], top_k: int
) -> None:
    payload = {
        "variant": variant,
        "dataset": "ezmed",
        "config": {"granularity": "chunk", "top_k": top_k, "gold": "two-judge"},
        "n_queries": len(per_query),
        "metrics": metrics,
        "per_query": per_query,
    }
    (RESULTS_DIR / f"ezmed_{variant}.json").write_text(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=RETRIEVAL_PATH)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth used in step 4.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
