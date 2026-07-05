"""Step 4b: build the Stage-2 gold standard with two LLM judges + tie-break.

Reads the pool from step 4, labels every (question, chunk) pair with Judge A
(gpt-5.5, Azure) and Judge B (DeepSeek, Foundry), resolves disagreements with an
o3 tie-break, and writes:
  - data/qa_dataset/qa_gold.jsonl        QAPairs with relevant_chunk_ids filled
  - results/ezmed_gold_report.json       kappa + agreement summary
  - results/ezmed_judge_labels.jsonl     per-pair labels (feeds human validation)

See local-docs/stage2-judge-protocol.md. Judges != gpt-4.1-mini (the HQ / question
generator) by design — the circularity guard.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from ezmed.evaluation.judge import RelevanceJudge, run_judging
from ezmed.evaluation.qa_dataset import export_jsonl, load_jsonl
from ezmed.ingestion.corpus import load_chunk_index
from ezmed.llm.client import LLMClient
from ezmed.logging import configure_logging
from ezmed.settings import settings

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
QA_PATH = Path("data/qa_dataset/qa_questions.jsonl")
POOL_PATH = Path("results/ezmed_pool.json")
GOLD_PATH = Path("data/qa_dataset/qa_gold.jsonl")
CHAT_CACHE_DIR = Path("data/processed/completions_cache")


def main() -> None:
    args = _parse_args()
    configure_logging()

    if not settings.azure_openai_key or not settings.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set")
    if not settings.deepseek_key or not settings.deepseek_base_url:
        raise RuntimeError("DEEPSEEK_KEY and DEEPSEEK_BASE_URL must be set (Judge B)")

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    qa_pairs = load_jsonl(args.qa)
    pools: dict[str, list[str]] = json.loads(args.pool.read_text())
    chunk_text, _ = load_chunk_index(RAW_DIR, limit=args.corpus_limit, seed=args.seed)
    _assert_pool_chunks_known(pools, chunk_text)

    # Judge A defaults to a reasoning deployment (gpt-5.5), which rejects a
    # non-default `temperature` — go through the reasoning path like the tie-break.
    judge_a = RelevanceJudge(
        _azure_client(settings.judge_a_deployment, reasoning=True), settings.judge_a_deployment
    )
    judge_b = RelevanceJudge(
        LLMClient.openai_compatible(
            deployment=settings.deepseek_deployment,
            api_key=settings.deepseek_key,
            base_url=settings.deepseek_base_url,
            cache_dir=CHAT_CACHE_DIR,
        ),
        settings.deepseek_deployment,
    )
    tiebreak = RelevanceJudge(
        _azure_client(settings.judge_tiebreak_deployment, reasoning=True),
        settings.judge_tiebreak_deployment,
    )
    logger.info(
        "judges: A=%s  B=%s  tie-break=%s",
        judge_a.name,
        judge_b.name,
        tiebreak.name,
    )

    updated, report = run_judging(
        qa_pairs, pools, chunk_text, judge_a, judge_b, tiebreak, workers=args.workers
    )

    export_jsonl(updated, args.output)
    (RESULTS_DIR / "ezmed_gold_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    with (RESULTS_DIR / "ezmed_judge_labels.jsonl").open("w", encoding="utf-8") as f:
        for label in report.labels:
            f.write(json.dumps(label.to_dict()) + "\n")

    empty_gold = sum(1 for qa in updated if not qa.relevant_chunk_ids)
    logger.info(
        "gold built: %d questions, %d with empty gold, inter-judge kappa=%.3f, "
        "agreement=%.1f%% (%d/%d tie-broken) in %.1fs",
        len(updated),
        empty_gold,
        report.inter_judge_kappa,
        report.agreement_rate * 100,
        report.n_disagree,
        report.n_pairs,
        time.time() - started,
    )


def _azure_client(deployment: str, reasoning: bool = False) -> LLMClient:
    return LLMClient(
        deployment=deployment,
        api_key=settings.azure_openai_key,
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        cache_dir=CHAT_CACHE_DIR,
        reasoning=reasoning,
    )


def _assert_pool_chunks_known(pools: dict[str, list[str]], chunk_text: dict[str, str]) -> None:
    missing = {cid for cids in pools.values() for cid in cids if cid not in chunk_text}
    if missing:
        raise RuntimeError(
            f"{len(missing)} pooled chunk_ids not in the reconstructed corpus — "
            "corpus-limit/seed likely differ from step 2/4."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=QA_PATH, help=f"QA JSONL ({QA_PATH}).")
    parser.add_argument("--pool", type=Path, default=POOL_PATH, help=f"Pool JSON ({POOL_PATH}).")
    parser.add_argument("--output", type=Path, default=GOLD_PATH, help=f"Gold JSONL ({GOLD_PATH}).")
    parser.add_argument(
        "--corpus-limit", type=int, default=100, help="Match step 2 --limit. Default 100."
    )
    parser.add_argument("--seed", type=int, default=42, help="Match step 2 --seed. Default 42.")
    parser.add_argument(
        "--workers",
        type=int,
        default=128,
        help="Concurrent judge calls per judge (A and B overlap).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
