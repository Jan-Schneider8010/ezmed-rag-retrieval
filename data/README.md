# data/

Local-only artifacts. Not committed to git. Per-run results live in `../results/`
at repo root and *are* committed.

| Folder        | Contents                                                    |
|---------------|-------------------------------------------------------------|
| `raw/`        | PubMed cardio papers as fetched (XML / JSON)           |
| `processed/`  | Parsed, cleaned, chunked text + embeddings cache            |
| `plaba/`      | PLABA dataset — Stage 1 external benchmark (CC BY 4.0). See https://osf.io/rnpmf/ |
| `qa_dataset/` | LLM-generated lay questions + manual gold-chunk annotations — Stage 2 |
