"""Langfuse client wrapper — no-op if credentials are not configured."""


class TrackingClient:
    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        self.enabled = bool(public_key and secret_key)
        self.host = host

    def trace_run(self, run_id: str, variant: str, metadata: dict) -> None:
        raise NotImplementedError

    def trace_llm_call(self, name: str, model: str, prompt: str, response: str) -> None:
        raise NotImplementedError
