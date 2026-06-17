from ezmed.evaluation.analysis import build_plaba_ablation, paired_significance


def test_paired_significance_degenerate_returns_nonsignificant() -> None:
    assert paired_significance([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == (0.0, 1.0)


def test_paired_significance_single_pair_is_nonsignificant() -> None:
    assert paired_significance([0.0], [1.0]) == (0.0, 1.0)


def test_paired_significance_detects_consistent_difference() -> None:
    base = [0.0] * 12
    cand = [1.0] * 12
    _, p = paired_significance(base, cand)
    assert p < 0.05


def _per_query(values: list[float]) -> list[dict]:
    return [
        {"qid": str(i), "recall_at_5": v, "recall_at_10": v, "mrr": 1.0, "ndcg_at_10": v}
        for i, v in enumerate(values)
    ]


def test_build_plaba_ablation_means_and_comparison_count() -> None:
    per_variant = {
        "baseline": _per_query([0.0] * 12),
        "qr_only": _per_query([1.0] * 12),
        "hq_only": _per_query([0.0] * 12),
        "both": _per_query([1.0] * 12),
    }
    out = build_plaba_ablation(per_variant)

    assert out["means"]["baseline"]["recall_at_10"] == 0.0
    assert out["means"]["qr_only"]["recall_at_10"] == 1.0
    assert out["n_comparisons"] == 6 * 2  # 6 pairs x 2 primary metrics
    assert out["bonferroni_alpha"] == 0.05 / 12

    bvq = next(
        c for c in out["comparisons"]
        if c["a"] == "baseline" and c["b"] == "qr_only" and c["metric"] == "recall_at_10"
    )
    assert bvq["p_value"] < 0.05
    assert bvq["significant"] is True
