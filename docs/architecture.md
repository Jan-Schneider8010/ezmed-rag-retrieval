# Architecture

This prototype implements the EZMed retrieval pipeline described in the exposé,
focused on the ablation study of LLM-based chunk enrichment (HQ) and query
rewriting (QR) for medical lay queries.

## Pipelines

```
PubMed OA ──► Parse ──► Chunk (1000/200) ──► [HQ enrichment]? ──► Embed ──► Qdrant
                                                                            │
Lay query ──► [Query rewrite]? ──► Embed ──► Vector search ◄────────────────┘
                                              │
                                              ▼
                                          Top-k chunks ──► (Answer LLM)
```

## Ablation variants

| Variant   | Ingestion              | Retrieval         |
|-----------|------------------------|-------------------|
| baseline  | plain chunks           | original query    |
| hq_only   | chunks + HQ            | original query    |
| qr_only   | plain chunks           | rewritten query   |
| both      | chunks + HQ            | rewritten query   |

Two Qdrant collections are populated — one for plain chunks, one for HQ-enriched
chunks. Variants reuse them rather than re-embedding the corpus four times.

## Storage

| System    | Holds                                                                |
|-----------|----------------------------------------------------------------------|
| Qdrant    | Chunk embeddings (3072-dim, text-embedding-3-large)                  |
| Postgres  | Paper metadata, chunks, QA pairs, run configurations, run results    |
| Langfuse  | (Optional) traces of LLM calls and runs                              |

## Evaluation

Primary metrics: Recall@5, Recall@10, MRR, NDCG@10.
Practical: ingestion / retrieval latency, token cost per paper and per query.
Statistics: paired Wilcoxon (or t-test) with Bonferroni correction across the
primary metrics, comparing each candidate variant to the baseline.

## Frontend (later)

A simple FastAPI service (`src/ezmed/api/main.py`) will expose `/search`. The
UI will be added once the pipeline produces stable results.
