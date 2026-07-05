"""Step 3: generate paper-level lay QA pairs (Stage 2).

Samples papers from the same corpus subset as step 2 (match --corpus-limit/--seed
so the source pmids line up with the ingested collections) and generates one lay
question per paper across the three prompting strategies. Writes a JSONL to
data/qa_dataset/ with relevant_chunk_ids empty — gold is established later by the
two-judge protocol (see local-docs/stage2-judge-protocol.md).
"""

import argparse
import logging
from pathlib import Path

from ezmed.evaluation.qa_dataset import QADatasetBuilder, export_jsonl
from ezmed.ingestion.corpus import load_corpus
from ezmed.llm.client import LLMClient
from ezmed.logging import configure_logging
from ezmed.settings import settings

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/qa_dataset/qa_questions.jsonl")
CHAT_CACHE_DIR = Path("data/processed/completions_cache")


def main() -> None:
    args = _parse_args()
    configure_logging()
    if not settings.azure_openai_key or not settings.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set")

    articles = load_corpus(RAW_DIR, limit=args.corpus_limit, seed=args.seed)

    chat_client = LLMClient(
        deployment=settings.azure_openai_chat_deployment,
        api_key=settings.azure_openai_key,
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        cache_dir=CHAT_CACHE_DIR,
    )

    builder = QADatasetBuilder(chat_client, n_questions=args.n_questions, seed=args.seed)
    pairs = builder.generate(list(articles.values()), workers=args.workers)
    export_jsonl(pairs, args.output)

    by_strategy: dict[str, int] = {}
    for pair in pairs:
        by_strategy[pair.prompting_strategy] = by_strategy.get(pair.prompting_strategy, 0) + 1
    logger.info("generated %d QA pairs %s", len(pairs), by_strategy)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-questions",
        type=int,
        default=100,
        help="Total questions (one per paper, over strategies); <= ingested papers. Default 100.",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=100,
        help="Corpus subsample size — MUST match step 2 --limit. Default 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for corpus subsample + paper sampling — match step 2. Default 42.",
    )
    parser.add_argument("--workers", type=int, default=32, help="Concurrent LLM calls.")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help=f"Output JSONL ({OUTPUT_PATH})."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
