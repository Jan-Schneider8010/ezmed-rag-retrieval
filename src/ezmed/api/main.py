"""FastAPI entrypoint — minimal /search endpoint for the upcoming frontend."""

from fastapi import FastAPI
from pydantic import BaseModel

from ezmed import __version__
from ezmed.schemas import Variant

app = FastAPI(title="EZMed RAG", version=__version__)


class SearchRequest(BaseModel):
    query: str
    variant: Variant = "both"
    top_k: int = 10


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    pmid: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    rewritten_query: str | None
    hits: list[SearchHit]
    latency_ms: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    raise NotImplementedError
