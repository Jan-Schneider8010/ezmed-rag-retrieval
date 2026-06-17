"""Qualitative error analysis of the PLABA ablation (file-based, no API calls).

Aligns per-query metrics across variants and pairs them with the original vs.
rewritten query text to understand where each technique helped or hurt.
"""

import json
from pathlib import Path
from statistics import mean

RESULTS = Path("results")
VARIANTS = ["baseline", "qr_only", "hq_only", "both"]
METRIC = "recall_at_10"


def _load(variant: str) -> dict[str, dict]:
    data = json.loads((RESULTS / f"plaba_{variant}.json").read_text())
    return {q["qid"]: q for q in data["per_query"]}


def main() -> None:
    pq = {v: _load(v) for v in VARIANTS}
    rewrites = json.loads((RESULTS / "plaba_rewritten_queries.json").read_text())
    qids = list(pq["baseline"])

    print("=" * 90)
    print(f"PER-QUERY DELTA vs baseline on {METRIC}  (n={len(qids)})")
    print("=" * 90)
    for v in ["qr_only", "hq_only", "both"]:
        deltas = [pq[v][q][METRIC] - pq["baseline"][q][METRIC] for q in qids]
        worse = [d for d in deltas if d < -1e-9]
        better = [d for d in deltas if d > 1e-9]
        print(f"\n{v:10s}  mean Δ={mean(deltas):+.4f}   "
              f"better={len(better):2d} (Σ{sum(better):+.3f})   "
              f"worse={len(worse):2d} (Σ{sum(worse):+.3f})   "
              f"same={len(qids) - len(better) - len(worse):2d}")

    # --- QR: where did rewriting cost recall? ---
    print("\n" + "=" * 90)
    print("QR_ONLY — queries where rewriting LOST recall@10 (sorted by damage)")
    print("=" * 90)
    qr_worse = sorted(
        ((pq["qr_only"][q][METRIC] - pq["baseline"][q][METRIC], q) for q in qids),
        key=lambda t: t[0],
    )
    for delta, q in qr_worse:
        if delta >= -1e-9:
            break
        r = rewrites[q]
        print(f"\n[qid {q}]  Δ={delta:+.2f}  "
              f"(baseline R@10={pq['baseline'][q][METRIC]:.2f} -> qr={pq['qr_only'][q][METRIC]:.2f})")
        print(f"  orig : {r['original']}")
        print(f"  rewr : {r['rewritten']}")

    # --- QR: where did it help? ---
    print("\n" + "=" * 90)
    print("QR_ONLY — queries where rewriting GAINED recall@10")
    print("=" * 90)
    for delta, q in sorted(qr_worse, key=lambda t: -t[0]):
        if delta <= 1e-9:
            break
        r = rewrites[q]
        print(f"\n[qid {q}]  Δ={delta:+.2f}")
        print(f"  orig : {r['original']}")
        print(f"  rewr : {r['rewritten']}")

    # --- Length-expansion vs damage ---
    print("\n" + "=" * 90)
    print("OVER-SPECIFICATION CHECK: rewrite length ratio vs recall@10 delta")
    print("=" * 90)
    rows = []
    for q in qids:
        r = rewrites[q]
        ratio = len(r["rewritten"]) / max(len(r["original"]), 1)
        rows.append((ratio, pq["qr_only"][q][METRIC] - pq["baseline"][q][METRIC], q))
    expanded = [d for ratio, d, _ in rows if ratio >= 1.8]
    modest = [d for ratio, d, _ in rows if ratio < 1.8]
    print(f"  strongly expanded rewrites (len ratio >= 1.8): n={len(expanded)}, mean Δ={mean(expanded):+.4f}")
    print(f"  modest rewrites           (len ratio <  1.8): n={len(modest)}, mean Δ={mean(modest):+.4f}")
    print("\n  biggest expansions:")
    for ratio, delta, q in sorted(rows, key=lambda t: -t[0])[:5]:
        r = rewrites[q]
        print(f"    [qid {q}] ratio={ratio:.1f} Δ={delta:+.2f}")
        print(f"      orig: {r['original']}")
        print(f"      rewr: {r['rewritten']}")

    # --- HQ: where did enrichment help most? ---
    print("\n" + "=" * 90)
    print("HQ_ONLY — queries where enrichment GAINED the most recall@10")
    print("=" * 90)
    hq_delta = sorted(
        ((pq["hq_only"][q][METRIC] - pq["baseline"][q][METRIC], q) for q in qids),
        key=lambda t: -t[0],
    )
    for delta, q in hq_delta[:8]:
        if delta <= 1e-9:
            break
        print(f"  [qid {q}] Δ={delta:+.2f}  orig: {rewrites[q]['original']}")


if __name__ == "__main__":
    main()
