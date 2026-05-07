.PHONY: help install up down logs shell test lint format ingest qa ablate evaluate clean

help:
	@echo "ezmed-rag-retrieval — common targets"
	@echo "  install     install python deps with dev extras"
	@echo "  up          start docker stack (qdrant, postgres, app)"
	@echo "  down        stop docker stack"
	@echo "  logs        tail docker logs"
	@echo "  shell       open a shell in the app container"
	@echo "  test        run pytest"
	@echo "  lint        ruff check"
	@echo "  format      ruff format"
	@echo "  ingest      run ingestion pipeline (script 02)"
	@echo "  qa          generate QA dataset (script 03)"
	@echo "  ablate      run ablation study (script 04)"
	@echo "  evaluate    compute metrics (script 05)"

install:
	pip install -e ".[dev]"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

shell:
	docker compose exec app bash

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

ingest:
	python scripts/02_ingest.py

qa:
	python scripts/03_generate_qa_dataset.py

ablate:
	python scripts/04_run_ablation.py

evaluate:
	python scripts/05_evaluate.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
