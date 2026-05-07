-- EZMed metadata + evaluation schema.
-- Vectors live in Qdrant; this DB stores paper metadata, chunks, QA pairs and run results.

CREATE TABLE IF NOT EXISTS papers (
    pmid          TEXT PRIMARY KEY,
    doi           TEXT,
    title         TEXT NOT NULL,
    journal       TEXT,
    authors       JSONB,
    mesh_terms    JSONB,
    published_at  DATE,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    section       TEXT,
    position      INTEGER NOT NULL,
    char_start    INTEGER NOT NULL,
    char_end      INTEGER NOT NULL,
    content       TEXT NOT NULL,
    hq_questions  JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_pmid ON chunks(pmid);

CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id              TEXT PRIMARY KEY,
    pmid               TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    question           TEXT NOT NULL,
    prompting_strategy TEXT NOT NULL,
    relevant_chunk_ids JSONB NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    variant       TEXT NOT NULL,
    config        JSONB NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS run_results (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    qa_id           TEXT NOT NULL REFERENCES qa_pairs(qa_id) ON DELETE CASCADE,
    retrieved_ids   JSONB NOT NULL,
    metrics         JSONB NOT NULL,
    latency_ms      INTEGER,
    tokens_used     INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_run_results_run ON run_results(run_id);
