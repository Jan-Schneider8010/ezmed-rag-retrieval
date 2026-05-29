from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ezmed.llm import client as client_module
from ezmed.llm.client import LLMClient


def _client(cache_dir: Path, deployment: str = "embed-dep") -> LLMClient:
    return LLMClient(
        deployment=deployment,
        api_key="k",
        endpoint="https://example.openai.azure.com/",
        api_version="2025-04-01-preview",
        cache_dir=cache_dir,
    )


@pytest.fixture
def mock_azure(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    instance = MagicMock()
    factory = MagicMock(return_value=instance)
    monkeypatch.setattr(client_module, "AzureOpenAI", factory)
    return instance


def _set_embeddings(mock_instance: MagicMock, vectors: list[list[float]]) -> None:
    mock_instance.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=v) for v in vectors]
    )


def test_requires_api_key() -> None:
    with pytest.raises(ValueError):
        LLMClient(
            deployment="d",
            api_key="",
            endpoint="https://example.openai.azure.com/",
            api_version="2025-04-01-preview",
        )


def test_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        LLMClient(
            deployment="d",
            api_key="k",
            endpoint="",
            api_version="2025-04-01-preview",
        )


def test_embed_returns_vectors_in_input_order(
    tmp_path: Path, mock_azure: MagicMock
) -> None:
    _set_embeddings(mock_azure, [[1.0], [2.0], [3.0]])
    assert _client(tmp_path).embed(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_embed_uses_deployment_as_model_param(
    tmp_path: Path, mock_azure: MagicMock
) -> None:
    _set_embeddings(mock_azure, [[1.0]])
    _client(tmp_path, deployment="my-dep").embed(["a"])
    assert mock_azure.embeddings.create.call_args.kwargs["model"] == "my-dep"


def test_cache_skips_api_on_second_call(
    tmp_path: Path, mock_azure: MagicMock
) -> None:
    _set_embeddings(mock_azure, [[1.0], [2.0]])
    client = _client(tmp_path)
    assert client.embed(["a", "b"]) == [[1.0], [2.0]]
    assert mock_azure.embeddings.create.call_count == 1

    assert client.embed(["a", "b"]) == [[1.0], [2.0]]
    assert mock_azure.embeddings.create.call_count == 1


def test_cache_only_requests_missing_texts(
    tmp_path: Path, mock_azure: MagicMock
) -> None:
    _set_embeddings(mock_azure, [[1.0], [2.0]])
    _client(tmp_path).embed(["a", "b"])

    _set_embeddings(mock_azure, [[3.0]])
    result = _client(tmp_path).embed(["a", "c"])
    assert result == [[1.0], [3.0]]
    assert mock_azure.embeddings.create.call_args.kwargs["input"] == ["c"]


def test_cache_key_includes_deployment(tmp_path: Path, mock_azure: MagicMock) -> None:
    _set_embeddings(mock_azure, [[1.0]])
    _client(tmp_path, deployment="d1").embed(["x"])

    _set_embeddings(mock_azure, [[9.0]])
    result = _client(tmp_path, deployment="d2").embed(["x"])
    assert result == [[9.0]]
    assert mock_azure.embeddings.create.call_count == 2


def test_empty_input_returns_empty_list(
    tmp_path: Path, mock_azure: MagicMock
) -> None:
    client = _client(tmp_path)
    assert client.embed([]) == []
    mock_azure.embeddings.create.assert_not_called()
