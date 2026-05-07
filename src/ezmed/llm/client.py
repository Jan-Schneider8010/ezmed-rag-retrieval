"""Thin OpenAI-compatible LLM client with retry + token accounting."""

from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    def complete(self, system: str, user: str, **kwargs: object) -> LLMResponse:
        raise NotImplementedError
