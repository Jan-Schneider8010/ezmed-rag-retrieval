"""Run a retrieval pipeline variant against the QA dataset and persist results."""

from ezmed.schemas import QAPair, RunMetrics, Variant


class AblationRunner:
    def __init__(self, variant: Variant) -> None:
        self.variant = variant

    def run(self, qa_pairs: list[QAPair]) -> RunMetrics:
        raise NotImplementedError
