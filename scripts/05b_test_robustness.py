"""Step 5b: check significance-test robustness — paired Wilcoxon vs. paired t-test.

Smucker et al. (CIKM 2007) advise against the Wilcoxon test for IR evaluation and
recommend the paired t-test / randomization tests instead. This step re-runs the six
pairwise Stage-2 comparisons with the paired t-test and checks where the
Bonferroni-corrected verdicts agree with the Wilcoxon results reported in step 5.
Reads the per-query metrics persisted by step 5 (results/ezmed_<variant>.json) and
writes results/ezmed_test_robustness.json.
"""

import argparse
import json
import logging
from pathlib import Path

from ezmed.evaluation.analysis import build_ablation
from ezmed.logging import configure_logging

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
VARIANTS = ("baseline", "qr_only", "hq_only", "both")


def main() -> None:
    args = _parse_args()
    configure_logging()

    suffix = f"_{args.tag}" if args.tag else ""
    per_variant = {
        variant: json.loads((RESULTS_DIR / f"ezmed_{variant}{suffix}.json").read_text())["per_query"]
        for variant in VARIANTS
    }

    ablations = {
        test: build_ablation(per_variant, id_key="qa_id", dataset="ezmed", test=test)
        for test in ("wilcoxon", "ttest")
    }

    comparisons = []
    n_flipped = 0
    pairs = zip(ablations["wilcoxon"]["comparisons"], ablations["ttest"]["comparisons"], strict=True)
    for wilc, tt in pairs:
        agree = wilc["significant"] == tt["significant"]
        n_flipped += not agree
        comparisons.append({
            "a": wilc["a"],
            "b": wilc["b"],
            "metric": wilc["metric"],
            "p_wilcoxon": wilc["p_value"],
            "p_ttest": tt["p_value"],
            "significant_wilcoxon": wilc["significant"],
            "significant_ttest": tt["significant"],
            "verdicts_agree": agree,
        })
        if not agree:
            logger.warning(
                "verdict flips: %s vs %s on %s (wilcoxon p=%.2e, ttest p=%.2e)",
                wilc["a"], wilc["b"], wilc["metric"], wilc["p_value"], tt["p_value"],
            )

    payload = {
        "dataset": "ezmed",
        "n_queries": ablations["wilcoxon"]["n_queries"],
        "bonferroni_alpha": ablations["wilcoxon"]["bonferroni_alpha"],
        "n_comparisons": len(comparisons),
        "n_verdicts_flipped": n_flipped,
        "comparisons": comparisons,
    }
    out = RESULTS_DIR / f"ezmed_test_robustness{suffix}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info(
        "%d/%d verdicts agree between Wilcoxon and paired t-test; wrote %s",
        len(comparisons) - n_flipped, len(comparisons), out,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="Suffix for default in/out paths, e.g. 'large'.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
