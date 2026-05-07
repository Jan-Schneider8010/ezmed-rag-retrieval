FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e ".[dev]"

COPY config ./config
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "ezmed.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
