"""Azure-OpenAI client with batched embeddings, chat completions, and disk cache."""

import hashlib
import logging
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path

from openai import APIError, AzureOpenAI, RateLimitError

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 100
_MAX_RETRIES = 4
_BASE_BACKOFF_S = 1.0


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    def __init__(
        self,
        deployment: str,
        api_key: str,
        endpoint: str,
        api_version: str,
        cache_dir: Path | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not endpoint:
            raise ValueError("endpoint is required")
        self.deployment = deployment
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [self._cache_get(t) for t in texts]
        missing_idx = [i for i, vec in enumerate(results) if vec is None]
        if missing_idx:
            logger.info(
                "embedding %d texts (cached: %d/%d)",
                len(missing_idx),
                len(texts) - len(missing_idx),
                len(texts),
            )
            for batch_start in range(0, len(missing_idx), _EMBED_BATCH_SIZE):
                batch_idx = missing_idx[batch_start : batch_start + _EMBED_BATCH_SIZE]
                batch_inputs = [texts[i] for i in batch_idx]
                vectors = self._embed_batch(batch_inputs)
                for i, vec in zip(batch_idx, vectors):
                    results[i] = vec
                    self._cache_put(texts[i], vec)

        return [vec for vec in results if vec is not None]

    def complete(
        self, system: str, user: str, temperature: float = 0.0, **kwargs: object
    ) -> LLMResponse:
        cached = self._completion_cache_get(system, user, temperature, kwargs)
        if cached is not None:
            return cached

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    **kwargs,
                )
                break
            except (RateLimitError, APIError) as err:
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = _BASE_BACKOFF_S * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "completion call failed (%s), retry %d/%d in %.1fs",
                    type(err).__name__,
                    attempt + 1,
                    _MAX_RETRIES - 1,
                    delay,
                )
                time.sleep(delay)
        else:
            raise RuntimeError("unreachable")

        usage = response.usage
        result = LLMResponse(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        self._completion_cache_put(system, user, temperature, kwargs, result)
        return result

    def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    model=self.deployment, input=inputs
                )
                return [d.embedding for d in response.data]
            except (RateLimitError, APIError) as err:
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = _BASE_BACKOFF_S * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "embedding call failed (%s), retry %d/%d in %.1fs",
                    type(err).__name__,
                    attempt + 1,
                    _MAX_RETRIES - 1,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def _cache_path(self, text: str) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(f"{self.deployment}\0{text}".encode()).hexdigest()
        return self._cache_dir / f"{digest}.pkl"

    def _cache_get(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if path is None or not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def _cache_put(self, text: str, vector: list[float]) -> None:
        path = self._cache_path(text)
        if path is None:
            return
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(vector, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def _completion_cache_path(
        self, system: str, user: str, temperature: float, kwargs: dict[str, object]
    ) -> Path | None:
        if self._cache_dir is None:
            return None
        params = repr(sorted(kwargs.items()))
        key = f"chat\0{self.deployment}\0{temperature}\0{params}\0{system}\0{user}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{digest}.pkl"

    def _completion_cache_get(
        self, system: str, user: str, temperature: float, kwargs: dict[str, object]
    ) -> "LLMResponse | None":
        path = self._completion_cache_path(system, user, temperature, kwargs)
        if path is None or not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def _completion_cache_put(
        self,
        system: str,
        user: str,
        temperature: float,
        kwargs: dict[str, object],
        response: "LLMResponse",
    ) -> None:
        path = self._completion_cache_path(system, user, temperature, kwargs)
        if path is None:
            return
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
