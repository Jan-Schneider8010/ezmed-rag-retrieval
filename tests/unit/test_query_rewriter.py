from unittest.mock import MagicMock

from ezmed.llm.client import LLMResponse
from ezmed.retrieval.query_rewriter import QueryRewriter


def test_rewrite_returns_stripped_text_and_total_tokens() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        text="  myocardial infarction  ", prompt_tokens=4, completion_tokens=6
    )
    rewritten, tokens = QueryRewriter(llm).rewrite("heart attack")
    assert rewritten == "myocardial infarction"
    assert tokens == 10


def test_rewrite_passes_lay_query_into_user_prompt() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text="x", prompt_tokens=1, completion_tokens=1)
    QueryRewriter(llm).rewrite("why does my chest hurt")
    system_arg, user_arg = llm.complete.call_args.args[:2]
    assert "why does my chest hurt" in user_arg
    assert isinstance(system_arg, str) and system_arg
