"""Postgres wrapper for papers, chunks, QA pairs and run results."""

from ezmed.schemas import Chunk, Paper, QAPair, RetrievalResult, Run


class MetadataStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def upsert_paper(self, paper: Paper) -> None:
        raise NotImplementedError

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def insert_qa_pairs(self, pairs: list[QAPair]) -> None:
        raise NotImplementedError

    def start_run(self, run: Run) -> None:
        raise NotImplementedError

    def record_result(self, run_id: str, result: RetrievalResult, metrics: dict) -> None:
        raise NotImplementedError

    def finish_run(self, run_id: str) -> None:
        raise NotImplementedError
