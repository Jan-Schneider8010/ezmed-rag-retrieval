"""Statistical analysis: paired Wilcoxon with Bonferroni correction (§4.5)."""

from itertools import combinations
from typing import Any

from scipy.stats import ttest_rel, wilcoxon

PLABA_VARIANTS = ("baseline", "qr_only", "hq_only", "both")
PRIMARY_METRICS = ("recall_at_10", "ndcg_at_10")
ALL_METRICS = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")


def paired_significance(
    baseline: list[float], candidate: list[float], test: str = "wilcoxon"
) -> tuple[float, float]:
    """Return (statistic, p_value) for a paired test over per-query scores.

    Degenerate input (all differences zero, or fewer than two pairs) yields
    (0.0, 1.0) instead of raising — common for MRR at PMID granularity.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired vectors must have equal length")
    diffs = [c - b for b, c in zip(baseline, candidate, strict=True)]
    if len(diffs) < 2 or all(d == 0 for d in diffs):
        return 0.0, 1.0
    if test == "wilcoxon":
        stat, p = wilcoxon(baseline, candidate)
    elif test == "ttest":
        stat, p = ttest_rel(baseline, candidate)
    else:
        raise ValueError(f"unknown test: {test!r}")
    return float(stat), float(p)


def build_ablation(
    per_variant: dict[str, list[dict[str, Any]]],
    id_key: str,
    dataset: str,
    alpha: float = 0.05,
    test: str = "wilcoxon",
) -> dict[str, Any]:
    """Assemble the ablation table + pairwise significance from per-query results.

    `per_variant` maps each variant to its per-query list (each item a dict with
    `id_key` and the metric keys). Vectors are aligned by id across variants; the
    six pairwise comparisons share a Bonferroni-corrected threshold.
    """
    variants = [v for v in PLABA_VARIANTS if v in per_variant]
    by_id = {v: {q[id_key]: q for q in per_variant[v]} for v in variants}
    common_ids = sorted(
        set.intersection(*(set(d) for d in by_id.values())),
        key=lambda x: (len(x), x),
    )

    means = {
        v: {m: _mean(by_id[v][i][m] for i in common_ids) for m in ALL_METRICS}
        for v in variants
    }

    pairs = list(combinations(variants, 2))
    n_comparisons = len(pairs) * len(PRIMARY_METRICS)
    bonferroni = alpha / n_comparisons if n_comparisons else alpha

    comparisons: list[dict[str, Any]] = []
    for a, b in pairs:
        for metric in PRIMARY_METRICS:
            va = [by_id[a][i][metric] for i in common_ids]
            vb = [by_id[b][i][metric] for i in common_ids]
            stat, p = paired_significance(va, vb, test=test)
            comparisons.append({
                "a": a,
                "b": b,
                "metric": metric,
                "mean_a": means[a][metric],
                "mean_b": means[b][metric],
                "statistic": stat,
                "p_value": p,
                "significant": p < bonferroni,
            })

    return {
        "dataset": dataset,
        "test": test,
        "n_queries": len(common_ids),
        "alpha": alpha,
        "n_comparisons": n_comparisons,
        "bonferroni_alpha": bonferroni,
        "primary_metrics": list(PRIMARY_METRICS),
        "means": means,
        "comparisons": comparisons,
    }


def build_plaba_ablation(
    per_variant: dict[str, list[dict[str, Any]]],
    alpha: float = 0.05,
    test: str = "wilcoxon",
) -> dict[str, Any]:
    """Stage-1 PLABA ablation (per-query dicts keyed by `qid`)."""
    return build_ablation(per_variant, id_key="qid", dataset="plaba", alpha=alpha, test=test)


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters. 1.0 = perfect, 0 = chance-level."""
    if len(a) != len(b):
        raise ValueError("rater vectors must have equal length")
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def _mean(values: Any) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
