"""Top-level Typer CLI: `ezmed <command>`."""

import typer

from ezmed.logging import configure_logging

app = typer.Typer(no_args_is_help=True, help="EZMed RAG retrieval prototype.")


@app.callback()
def _root() -> None:
    configure_logging()


@app.command()
def collect() -> None:
    """Download the PubMed corpus."""
    raise NotImplementedError


@app.command()
def ingest(variant: str = "baseline") -> None:
    """Run the ingestion pipeline for a given variant."""
    raise NotImplementedError


@app.command("generate-qa")
def generate_qa() -> None:
    """Generate the lay-query QA evaluation dataset."""
    raise NotImplementedError


@app.command()
def ablate() -> None:
    """Run the ablation study across all variants."""
    raise NotImplementedError


@app.command()
def evaluate(run_id: str) -> None:
    """Compute metrics for a completed run."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
