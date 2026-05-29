# ezmed-rag-retrieval

Bachelor thesis prototype — *Improving Retrieval Quality for Medical Lay Queries
through LLM-Based Chunk Enrichment and Query Rewriting* (Jan Schneider, TU Berlin,
April 2026).

A RAG pipeline for ~1.000 open-access PubMed papers in cardiology.
Implements four pipeline variants in an ablation study:

| Variant   | Ingestion              | Retrieval         |
|-----------|------------------------|-------------------|
| baseline  | plain chunks           | original query    |
| hq_only   | chunks + hypothetical lay questions | original query |
| qr_only   | plain chunks           | LLM query rewriting |
| both      | chunks + HQ            | LLM query rewriting |

Evaluation: Recall@5, Recall@10, MRR, NDCG@10, plus latency and token cost.

## Stack

Python 3.11 · Docker · Qdrant (vector DB) · Postgres (metadata + results) ·
OpenAI `text-embedding-3-large` · Langfuse (optional tracing) · FastAPI
(for the upcoming frontend).

## Layout

```
ezmed-rag-retrieval/
├── config/                 # settings + ablation variants (pipelines.yaml)
├── docker/                 # app Dockerfile, postgres init, qdrant config
├── docker-compose.yml      # local stack: qdrant + postgres + app
├── src/ezmed/
│   ├── ingestion/          # pubmed → parse → chunk → HQ → embed
│   ├── retrieval/          # rewrite → embed → search → answer
│   ├── storage/            # qdrant + postgres wrappers
│   ├── llm/                # openai client + prompt loader
│   ├── evaluation/         # qa dataset, metrics, ablation, stats
│   ├── tracking/           # langfuse client
│   ├── api/                # fastapi service for the frontend
│   ├── schemas.py          # Pydantic models shared across the codebase
│   └── cli.py              # `ezmed <command>`
├── prompts/                # plain-text prompt artefacts (system.md + user.md per prompt)
├── scripts/                # numbered end-to-end runs (01..05)
├── tests/                  # unit + integration tests (pytest)
├── data/                   # raw/processed/qa_dataset (gitignored)
├── results/                # per-run metrics + stats output (committed)
├── notebooks/              # exploration + results analysis
└── docs/architecture.md    # design notes
```

## Quickstart

```bash
cp .env.example .env             # fill in AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT
uv sync --all-extras
uv run nbstripout --install      # one-time: clean notebook diffs on commit
docker compose up -d qdrant
```

## End-to-end run

```bash
python scripts/01_collect_corpus.py        # ~1000 PubMed OA papers
python scripts/02_ingest.py                # chunk + (HQ) + embed → Qdrant
python scripts/03_generate_qa_dataset.py   # 100–150 lay QA pairs
# manual gold-chunk annotation
python scripts/04_run_ablation.py          # all 4 variants
python scripts/05_evaluate.py              # Recall@k, MRR, NDCG, stats
```

## Development

```bash
make test
make lint
make format
```

## License

MIT — see `LICENSE`.
