"""Pydantic models shared across ingestion, retrieval and evaluation."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Variant = Literal["baseline", "hq_only", "qr_only", "both"]
PromptingStrategy = Literal["naive_lay", "symptom_based", "everyday_paraphrase"]


class Paper(BaseModel):
    pmid: str
    doi: str | None = None
    title: str
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    published_at: date | None = None
    full_text: str | None = None


class Section(BaseModel):
    title: str
    text: str


class ParsedArticle(BaseModel):
    pmid: str
    pmcid: str | None = None
    doi: str | None = None
    title: str
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: list[Section] = Field(default_factory=list)
    body: list[Section] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    pmid: str
    section: str | None
    position: int
    char_start: int
    char_end: int
    content: str
    hq_questions: list[str] = Field(default_factory=list)


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    embedding: list[float]


class QAPair(BaseModel):
    qa_id: str
    pmid: str
    question: str
    prompting_strategy: PromptingStrategy
    relevant_chunk_ids: list[str]


class RetrievalResult(BaseModel):
    qa_id: str
    retrieved_ids: list[str]
    scores: list[float]
    rewritten_query: str | None = None
    latency_ms: int
    tokens_used: int = 0


class RunMetrics(BaseModel):
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    avg_latency_ms: float
    avg_tokens: float


class Run(BaseModel):
    run_id: str
    variant: Variant
    config: dict
    started_at: datetime
    finished_at: datetime | None = None
    metrics: RunMetrics | None = None
