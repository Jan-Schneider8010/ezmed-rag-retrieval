"""QA-dataset construction: paper-level LLM-generated lay questions (Stage 2).

One lay question per sampled paper, generated from title + abstract across three
prompting strategies. Questions are deliberately NOT chunk-seeded: gold relevance
is established entirely by the two-judge protocol over a pool (see
local-docs/stage2-judge-protocol.md), which reduces the query<->HQ circularity and
avoids a ceiling effect. `relevant_chunk_ids` starts empty; `pmid` records the
source paper so its chunks can be added to the judging pool. Sampling is
deterministic (seeded) so a dataset re-generates identically."""

import logging
from pathlib import Path
from random import Random

from ezmed.llm.client import LLMClient
from ezmed.llm.prompts import Prompt, load_prompt, load_qa_strategy
from ezmed.schemas import ParsedArticle, PromptingStrategy, QAPair

logger = logging.getLogger(__name__)

STRATEGIES: tuple[PromptingStrategy, ...] = (
    "naive_lay",
    "symptom_based",
    "everyday_paraphrase",
)
MIN_ABSTRACT_CHARS = 200


def abstract_text(article: ParsedArticle) -> str:
    return "\n".join(s.text for s in article.abstract).strip()


class QADatasetBuilder:
    """Sample papers and generate one lay question per paper."""

    def __init__(self, llm: LLMClient, n_questions: int, seed: int = 42) -> None:
        self.llm = llm
        self.n_questions = n_questions
        self.seed = seed

    def sample_papers(
        self, articles: list[ParsedArticle]
    ) -> list[tuple[ParsedArticle, PromptingStrategy]]:
        """Deterministically pick papers with a usable abstract + assign a strategy."""
        eligible = [a for a in articles if len(abstract_text(a)) >= MIN_ABSTRACT_CHARS]
        Random(self.seed).shuffle(eligible)
        picked = eligible[: self.n_questions]
        if len(picked) < self.n_questions:
            logger.warning(
                "only %d papers with abstract for %d requested questions",
                len(picked), self.n_questions,
            )
        return [(a, STRATEGIES[i % len(STRATEGIES)]) for i, a in enumerate(picked)]

    def generate(self, articles: list[ParsedArticle], workers: int = 1) -> list[QAPair]:
        prompt = load_prompt("qa_generation")
        sampled = self.sample_papers(articles)
        logger.info("generating %d questions (workers=%d)", len(sampled), workers)

        def one(pair: tuple[ParsedArticle, PromptingStrategy]) -> str:
            article, strategy = pair
            return self._generate_one(prompt, article, strategy)

        if workers > 1 and len(sampled) > 1:
            from ezmed.ingestion.pipeline import parallel_map

            questions = parallel_map(one, sampled, workers, "QA gen")
        else:
            questions = [one(p) for p in sampled]

        return [
            QAPair(
                qa_id=f"qa_{i:04d}",
                pmid=article.pmid,
                question=question,
                prompting_strategy=strategy,
                relevant_chunk_ids=[],
            )
            for i, ((article, strategy), question) in enumerate(
                zip(sampled, questions, strict=True)
            )
        ]

    def _generate_one(
        self, prompt: Prompt, article: ParsedArticle, strategy: PromptingStrategy
    ) -> str:
        response = self.llm.complete(
            prompt.system,
            prompt.render_user(
                title=article.title,
                abstract=abstract_text(article),
                style_instruction=load_qa_strategy(strategy),
            ),
        )
        return _clean_question(response.text)


def _clean_question(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip().strip('"').strip()
        if cleaned:
            return cleaned
    return text.strip()


def export_jsonl(pairs: list[QAPair], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")
    logger.info("wrote %d QA pairs to %s", len(pairs), path)
    return path


def load_jsonl(path: Path) -> list[QAPair]:
    with path.open(encoding="utf-8") as f:
        return [QAPair.model_validate_json(line) for line in f if line.strip()]
