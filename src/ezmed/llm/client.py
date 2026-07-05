"""Azure-OpenAI client with batched embeddings, chat completions, and disk cache."""

import hashlib
import logging
import os
import pickle
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import APIError, AzureOpenAI, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


def _atomic_write(path: Path, obj: object) -> None:
    """Pickle obj to path via a uniquely-named temp file + atomic replace.

    The unique temp name is essential: several threads may cache the same key
    (identical query/completion) concurrently. A shared temp name races on
    os.replace — one thread's replace pulls the file out from under another's."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


_EMBED_BATCH_SIZE = 100
_EMBED_WORKERS = 32
_MAX_RETRIES = 4
_BASE_BACKOFF_S = 1.0

# The openai SDK's default httpx pool caps at 100 connections / 20 keepalive,
# which throttles high --workers runs before the (very high) TPM limit bites.
# Size the pool for the concurrency the account can actually sustain.
_MAX_CONNECTIONS = 512

# Read timeout: a request that stalls server-side under a concurrent burst must
# fail fast so the retry loop can reissue it on a fresh connection, rather than
# blocking a worker for minutes. 120s is ample headroom over observed latency
# (embeddings <2s; reasoning judges a few seconds), yet self-heals a stall in 2m.
_READ_TIMEOUT_S = 120.0


def _http_client() -> httpx.Client:
    return httpx.Client(
        limits=httpx.Limits(
            max_connections=_MAX_CONNECTIONS,
            max_keepalive_connections=_MAX_CONNECTIONS,
        ),
        timeout=httpx.Timeout(_READ_TIMEOUT_S, connect=10.0),
    )


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
        api_key: str = "",
        endpoint: str = "",
        api_version: str = "",
        cache_dir: Path | None = None,
        *,
        client: AzureOpenAI | OpenAI | None = None,
        reasoning: bool = False,
    ) -> None:
        self.deployment = deployment
        self.reasoning = reasoning
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise ValueError("api_key is required")
            if not endpoint:
                raise ValueError("endpoint is required")
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
                http_client=_http_client(),
            )
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def openai_compatible(
        cls,
        deployment: str,
        api_key: str,
        base_url: str,
        cache_dir: Path | None = None,
        reasoning: bool = False,
    ) -> "LLMClient":
        """Client for an OpenAI-compatible endpoint (DeepSeek on Azure AI Foundry)."""
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")
        return cls(
            deployment=deployment,
            cache_dir=cache_dir,
            client=OpenAI(api_key=api_key, base_url=base_url, http_client=_http_client()),
            reasoning=reasoning,
        )

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
            batches = [
                missing_idx[s : s + _EMBED_BATCH_SIZE]
                for s in range(0, len(missing_idx), _EMBED_BATCH_SIZE)
            ]

            def run(batch_idx: list[int]) -> tuple[list[int], list[list[float]]]:
                return batch_idx, self._embed_batch([texts[i] for i in batch_idx])

            if len(batches) <= 1:
                completed = [run(b) for b in batches]
            else:
                with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
                    completed = [
                        f.result() for f in as_completed(pool.submit(run, b) for b in batches)
                    ]

            for batch_idx, vectors in completed:
                for i, vec in zip(batch_idx, vectors, strict=True):
                    results[i] = vec
                    self._cache_put(texts[i], vec)

        return [vec for vec in results if vec is not None]

    def complete(
        self, system: str, user: str, temperature: float = 0.0, **kwargs: object
    ) -> LLMResponse:
        cached = self._completion_cache_get(system, user, temperature, kwargs)
        if cached is not None:
            return cached

        params = dict(kwargs)
        if self.reasoning:
            # Reasoning deployments (o3, gpt-5 reasoning) reject `temperature` and
            # rename max_tokens -> max_completion_tokens.
            if "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens")
        else:
            params["temperature"] = temperature

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **params,
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
                response = self._client.embeddings.create(model=self.deployment, input=inputs)
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
        _atomic_write(path, vector)

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
        _atomic_write(path, response)
