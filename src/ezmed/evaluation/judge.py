"""Two-LLM-judge relevance labeling for the Stage-2 gold standard.

Given the candidate pool per question (source-paper chunks + pooled retrieval
hits), two independent judges from different model families label each
(question, chunk) pair as relevant/not. Agreement is reported as Cohen's kappa;
disagreements are resolved by a reasoning-model tie-break. The resulting gold
populates `QAPair.relevant_chunk_ids`. See local-docs/stage2-judge-protocol.md."""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from ezmed.evaluation.analysis import cohen_kappa
from ezmed.ingestion.pipeline import parallel_map
from ezmed.llm.prompts import load_prompt
from ezmed.schemas import QAPair, Variant

logger = logging.getLogger(__name__)

# Pool depth per variant. Must be >= the largest metric cutoff (Recall@10 /
# NDCG@10): chunks a variant ranks 1..10 all get judged, so nothing in a metric
# window is silently treated as non-relevant.
POOL_DEPTH = 10


class ChatClient(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> object: ...


@dataclass(frozen=True)
class PoolItem:
    qa_id: str
    chunk_id: str
    question: str
    chunk_text: str


@dataclass(frozen=True)
class PairLabel:
    qa_id: str
    chunk_id: str
    judge_a: bool
    judge_b: bool
    final: bool

    def to_dict(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "chunk_id": self.chunk_id,
            "judge_a": int(self.judge_a),
            "judge_b": int(self.judge_b),
            "final": int(self.final),
        }


@dataclass
class JudgingReport:
    n_pairs: int
    n_agree: int
    n_disagree: int
    inter_judge_kappa: float
    n_relevant: int
    per_qa_gold_size: dict[str, int] = field(default_factory=dict)
    labels: list[PairLabel] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        return self.n_agree / self.n_pairs if self.n_pairs else 0.0

    def to_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "n_agree": self.n_agree,
            "n_disagree": self.n_disagree,
            "agreement_rate": round(self.agreement_rate, 4),
            "inter_judge_kappa": round(self.inter_judge_kappa, 4),
            "n_relevant": self.n_relevant,
            "per_qa_gold_size": self.per_qa_gold_size,
        }


class RelevanceJudge:
    """Wraps a chat client + the relevance rubric; returns a binary label."""

    def __init__(self, client: ChatClient, name: str) -> None:
        self.client = client
        self.name = name
        self._prompt = load_prompt("relevance_judge")

    def label(self, item: PoolItem) -> bool:
        response = self.client.complete(
            self._prompt.system,
            self._prompt.render_user(question=item.question, chunk=item.chunk_text),
        )
        return _parse_label(response.text)


def _parse_label(text: str) -> bool:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return bool(int(json.loads(match.group(0)).get("relevant", 0)))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    lowered = text.lower()
    if re.search(r"relevant\D{0,4}1", lowered) or lowered.strip().startswith("1"):
        return True
    return False


def build_pool(
    qa: QAPair,
    retrieved: dict[Variant, list[str]],
    source_chunk_ids: list[str],
    k: int = POOL_DEPTH,
) -> list[str]:
    """Pool = source-paper chunks + union of top-k retrieval hits across variants.

    Deduplicated: a chunk retrieved by several variants (or twice by one), or a
    source-paper chunk that was also retrieved, appears exactly once. Deterministic
    order: source chunks first (their given order), then remaining pooled chunks
    sorted, so a re-run judges the same pool identically."""
    pool = list(dict.fromkeys(source_chunk_ids))
    seen = set(pool)
    extra: set[str] = set()
    for ids in retrieved.values():
        extra.update(ids[:k])
    for chunk_id in sorted(extra - seen):
        pool.append(chunk_id)
    return pool


def run_judging(
    qa_pairs: list[QAPair],
    pools: dict[str, list[str]],
    chunk_text: dict[str, str],
    judge_a: RelevanceJudge,
    judge_b: RelevanceJudge,
    tiebreak: RelevanceJudge,
    workers: int = 8,
) -> tuple[list[QAPair], JudgingReport]:
    """Label every pooled (question, chunk) pair with both judges, adjudicate
    disagreements via the tie-break judge, and return QAPairs with gold filled in."""
    questions = {qa.qa_id: qa.question for qa in qa_pairs}
    items = [
        PoolItem(qa_id, cid, questions[qa_id], chunk_text[cid])
        for qa_id, cids in pools.items()
        for cid in cids
    ]
    if not items:
        return qa_pairs, JudgingReport(0, 0, 0, 0.0, 0)

    # Judge A (Azure) and Judge B (Foundry) hit separate endpoints — no shared
    # rate limit — so overlap them: label phase = max(A, B) instead of A + B.
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(parallel_map, judge_a.label, items, workers, f"judge {judge_a.name}")
        fut_b = pool.submit(parallel_map, judge_b.label, items, workers, f"judge {judge_b.name}")
        labels_a = fut_a.result()
        labels_b = fut_b.result()
    kappa = cohen_kappa(labels_a, labels_b)

    disagree_idx = [i for i, (a, b) in enumerate(zip(labels_a, labels_b, strict=True)) if a != b]
    tb_labels = parallel_map(
        tiebreak.label, [items[i] for i in disagree_idx], workers, f"tiebreak {tiebreak.name}"
    )
    resolved = dict(zip(disagree_idx, tb_labels, strict=True))

    final = [resolved[i] if i in resolved else labels_a[i] for i in range(len(items))]

    gold: dict[str, list[str]] = {qa.qa_id: [] for qa in qa_pairs}
    for item, is_relevant in zip(items, final, strict=True):
        if is_relevant:
            gold[item.qa_id].append(item.chunk_id)

    updated = [qa.model_copy(update={"relevant_chunk_ids": gold[qa.qa_id]}) for qa in qa_pairs]
    labels = [
        PairLabel(item.qa_id, item.chunk_id, la, lb, fin)
        for item, la, lb, fin in zip(items, labels_a, labels_b, final, strict=True)
    ]
    report = JudgingReport(
        n_pairs=len(items),
        n_agree=len(items) - len(disagree_idx),
        n_disagree=len(disagree_idx),
        inter_judge_kappa=kappa,
        n_relevant=sum(final),
        per_qa_gold_size={qa_id: len(cids) for qa_id, cids in gold.items()},
        labels=labels,
    )
    logger.info(
        "judged %d pairs: agree=%.1f%% kappa=%.3f relevant=%d",
        report.n_pairs,
        report.agreement_rate * 100,
        kappa,
        report.n_relevant,
    )
    return updated, report
