# results/

Per-run retrieval results and statistics for the ablation. Every number that
ends up in the thesis PDF must trace back to a file here.

## File naming

Files are written by `scripts/02_run_baseline_plaba.py` and
`scripts/03_run_plaba_ablation.py`. The optional `--tag <name>` flag appends a
`_<name>` suffix to every output file (and to the Qdrant collection names), so
several runs can coexist without overwriting each other.

| Suffix     | Run                              | Embedding model          |
|------------|----------------------------------|--------------------------|
| *(none)*   | primary run (no `--tag`)         | `text-embedding-3-large` |
| `_large`   | `--tag large`                    | `text-embedding-3-large` |
| `_small`   | `--tag small`                    | `text-embedding-3-small` |

The **no-suffix** files are the canonical run the analysis notebook
(`notebooks/02_results_analysis.ipynb`) reads. The `_small` / `_large` files
are an embedding-model robustness check — same pipeline, swapped embedding
model — used to confirm the ablation pattern is not an artefact of one model.

## Files per run

For each `<suffix>`:

- `plaba_<variant><suffix>.json` — one per variant (`baseline`, `qr_only`,
  `hq_only`, `both`): config, aggregate metrics, and per-query scores.
- `plaba_ablation<suffix>.json` — cross-variant means + pairwise paired
  Wilcoxon comparisons with Bonferroni correction.
- `plaba_rewritten_queries<suffix>.json` — lay → medical query rewrites
  (for qualitative error analysis).
- `plaba_hq_questions<suffix>.json` — raw per-chunk hypothetical questions.
  **Not committed** (~2 MB each, regenerable from the LLM cache, not a thesis
  number) — see the `.gitignore` rule.
