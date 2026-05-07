"""Typed application settings, loaded from environment / .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM / embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    llm_hq_model: str = "gpt-4o-mini"
    llm_query_rewrite_model: str = "gpt-4o-mini"
    llm_answer_model: str = "gpt-4o"

    # PubMed
    pubmed_email: str = ""
    pubmed_api_key: str = ""
    pubmed_domain: str = "cardiology"
    pubmed_corpus_size: int = 1000

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ezmed_chunks"

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ezmed"
    postgres_user: str = "ezmed"
    postgres_password: str = "ezmed"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    hq_per_chunk: int = 4

    # App
    log_level: str = "INFO"
    data_dir: Path = Field(default=Path("./data"))
    prompts_dir: Path = Field(default=Path("./prompts"))
    results_dir: Path = Field(default=Path("./results"))

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
