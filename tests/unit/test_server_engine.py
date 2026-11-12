"""Native protocol semantics using test-only token IDs and monotonic timestamps."""

import copy
from typing import Any

import pytest

from backend.engines.http_stream import SSEEvent, StreamResponse
from backend.engines.llama_server import analyze_stream


def native_fixture(count: int = 3, target: int = 3) -> tuple[StreamResponse, dict[str, Any]]:
    request = {
        "prompt": [1, 2, 3],
        "n_predict": target,
        "seed": 42,
        "temperature": 0.0,
        "ignore_eos": True,
        "samplers": ["temperature"],
        "cache_prompt": False,
    }
    final = {
        "stop": True,
        "content": "",
        "tokens": [],
        "tokens_evaluated": 3,
        "tokens_predicted": count,
        "tokens_cached": 999,
        "truncated": False,
        "stop_type": "limit" if count == target else "eos",
        "generation_settings": {
            k: request[k] for k in ("n_predict", "seed", "temperature", "ignore_eos", "samplers")
        },
        "timings": {
            "cache_n": 0,
            "prompt_n": 3,
            "predicted_n": count,
            "prompt_ms": 3.0,
            "prompt_per_second": 1000.0,
            "predicted_ms": max(0, count - 1) * 10.0,
            "predicted_per_second": 100.0 if count >= 2 else 0.0,
        },
    }
    events = [SSEEvent(None, 110_000_000, "null")]
    events += [
        SSEEvent(
            {"stop": False, "tokens": [100 + i], "tokens_predicted": i + 1, "content": str(i)},
            120_000_000 + i * 10_000_000,
            "TEST ONLY",
        )
        for i in range(count)
    ]
    events.append(SSEEvent(final, 145_000_000, "TEST ONLY final"))
    return StreamResponse(events, 100_000_000, 145_000_000), request


def test_native_counts_and_first_token_formula_and_header_exclusion() -> None:
    stream, request = native_fixture()
    sample = analyze_stream(stream, request, 2)
    assert sample["trial_index"] == 2
    assert sample["ttft_ms"] == 20.0
    assert sample["request_latency_ms"] == 45.0
    assert sample["mean_inter_token_latency_ms"] == 10.0
    assert sample["median_inter_token_latency_ms"] == 10.0
    assert sample["decode_tokens_per_second"] == 100.0
    assert sample["cache_reused_tokens"] == 0  # tokens_cached=999 is final slot state.
    assert sample["generated_tokens"] == 3
    assert sample["early_stop"] is False
    assert sample["output"] == "012"
    assert sample["events"][1]["arrival_ns"] == 20_000_000


@pytest.mark.parametrize("count", [0, 1])
def test_early_and_short_outputs_keep_counts_and_unavailable_decode(count: int) -> None:
    stream, request = native_fixture(count=count)
    sample = analyze_stream(stream, request)
    assert sample["generated_tokens"] == count
    assert sample["early_stop"] is True
    assert sample["decode_tokens_per_second"] is None
    assert sample["mean_inter_token_latency_ms"] is None
    if count == 0:
        assert sample["ttft_ms"] is None


@pytest.mark.parametrize("kind", ["coverage_gap", "multiple_ids", "coalesced"])
def test_ambiguous_token_events_leave_intervals_unavailable(kind: str) -> None:
    stream, request = native_fixture()
    events = list(stream.events)
    if kind == "coverage_gap":
        del events[1]
    elif kind == "multiple_ids":
        events[2].data["tokens"] = [100, 101]
        del events[1]
    else:
        events[2] = SSEEvent(events[2].data, events[1].arrival_ns, "coalesced TEST ONLY")
    sample = analyze_stream(StreamResponse(events, stream.started_ns, stream.completed_ns), request)
    assert sample["generated_tokens"] == 3
    assert sample["mean_inter_token_latency_ms"] is None
    assert sample["median_inter_token_latency_ms"] is None
    assert sample["warnings"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("tokens_evaluated", 4),
        ("tokens_predicted", True),
        ("truncated", True),
        ("stop_type", "unknown"),
        ("stop", False),
    ],
)
def test_bad_terminal_evidence_is_rejected(key: str, value: Any) -> None:
    stream, request = native_fixture()
    stream.events[-1].data[key] = value
    with pytest.raises(ValueError):
        analyze_stream(stream, request)


def test_cache_reuse_is_rejected_even_with_consistent_total_tokens() -> None:
    stream, request = native_fixture()
    stream.events[-1].data["timings"].update(cache_n=1, prompt_n=2)
    with pytest.raises(ValueError, match="prefix-cache"):
        analyze_stream(stream, request)


def test_decode_rate_must_exclude_first_token() -> None:
    stream, request = native_fixture()
    stream.events[-1].data["timings"]["predicted_per_second"] = 150.0
    with pytest.raises(ValueError, match="decode timing"):
        analyze_stream(stream, request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ignore_eos", False),
        ("seed", True),
        ("n_predict", True),
        ("temperature", float("nan")),
        ("samplers", ["top_k"]),
    ],
)
def test_observed_sampling_settings_must_match_request(field: str, value: Any) -> None:
    stream, request = native_fixture()
    stream.events[-1].data["generation_settings"][field] = value
    with pytest.raises(ValueError):
        analyze_stream(stream, request)


def test_raw_request_and_terminal_evidence_are_retained() -> None:
    stream, request = native_fixture()
    expected = copy.deepcopy(stream.events[-1].data)
    sample = analyze_stream(stream, request)
    assert sample["request"] == request
    assert sample["response"] == expected
