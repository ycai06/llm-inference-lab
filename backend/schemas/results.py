"""Normalized request measurements; raw protocol evidence remains attached."""

from typing import Any, TypedDict


class RequestSample(TypedDict):
    trial_index: int
    request_latency_ms: float
    ttft_ms: float | None
    prompt_tokens: int
    prompt_processed_tokens: int
    generated_tokens: int
    target_generated_tokens: int
    prefill_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    mean_inter_token_latency_ms: float | None
    median_inter_token_latency_ms: float | None
    cache_reused_tokens: int
    early_stop: bool
    finish_reason: str
    output: str
    events: list[dict[str, Any]]
    request: dict[str, Any]
    response: dict[str, Any]
    warnings: list[str]
