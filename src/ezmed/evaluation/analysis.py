"""Statistical analysis: paired t-test / Wilcoxon with Bonferroni correction (§4.5)."""

from ezmed.schemas import RunMetrics


def paired_significance(
    baseline: list[float], candidate: list[float], test: str = "wilcoxon"
) -> tuple[float, float]:
    """Return (statistic, p_value)."""
    raise NotImplementedError


def compare_runs(
    baseline_metrics: RunMetrics, candidate_metrics: RunMetrics
) -> dict[str, float]:
    """Compute deltas and Bonferroni-corrected p-values per primary metric."""
    raise NotImplementedError
