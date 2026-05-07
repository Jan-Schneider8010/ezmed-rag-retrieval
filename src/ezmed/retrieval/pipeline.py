"""Retrieval orchestrator: dispatches one of the four ablation variants."""

from ezmed.schemas import RetrievalResult, Variant


class RetrievalPipeline:
    def __init__(self, variant: Variant, top_k: int) -> None:
        self.variant = variant
        self.top_k = top_k

    def run(self, qa_id: str, query: str) -> RetrievalResult:
        raise NotImplementedError
