"""QA-dataset construction: LLM-generated lay questions + manual gold-chunk annotation."""

from pathlib import Path

from ezmed.schemas import Paper, QAPair


class QADatasetBuilder:
    def __init__(self, output_dir: Path, k_per_paper: int) -> None:
        self.output_dir = output_dir
        self.k_per_paper = k_per_paper

    def generate_candidates(self, papers: list[Paper]) -> list[QAPair]:
        """LLM-generate lay questions across the three prompting strategies."""
        raise NotImplementedError

    def export_for_annotation(self, pairs: list[QAPair]) -> Path:
        """Write a CSV/JSONL the annotator can fill in with relevant_chunk_ids."""
        raise NotImplementedError

    def load(self, path: Path) -> list[QAPair]:
        raise NotImplementedError
