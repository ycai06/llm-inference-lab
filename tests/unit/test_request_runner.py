"""Warm-request orchestration with an in-process test backend and a tiny GGUF header."""

from __future__ import annotations

import signal
import struct
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from backend.benchmarks import request_runner
from backend.schemas.request_config import RequestConfig
from backend.storage.database import Store


class TestRequestSession:
    __test__ = False

    def __init__(self, engine: TestRequestEngine) -> None:
        self.engine = engine
        self.metadata: dict[str, Any] = {
            "TEST_ONLY": True,
            "startup_duration_ms": 2000.0,
            "closed": False,
        }

    def __enter__(self) -> TestRequestSession:
        self.engine.enters += 1
        return self

    def __exit__(self, *args: Any) -> None:
        self.engine.exits += 1
        self.metadata.update(closed=True, returncode=0, stderr="TEST ONLY final child log")

    def tokenize(
        self,
        text: str,
        add_special: bool = True,
        parse_special: bool = False,
        timeout_s: float = 30,
    ) -> list[int]:
        assert text == self.engine.fixture_text
        assert add_special is True
        assert parse_special is False
        assert timeout_s > 0
        self.engine.tokenize_calls += 1
        return list(range(self.engine.fixture_token_count))

    def complete(
        self,
        prompt_tokens: list[int],
        generation_tokens: int,
        seed: int,
        ignore_eos: bool,
        cache_prompt: bool,
        timeout_s: float,
        cancel: threading.Event,
        trial_index: int = 0,
    ) -> dict[str, Any]:
        assert seed == 42
        assert cache_prompt is False
        assert timeout_s > 0
        self.engine.requests.append(
            {
                "prompt": prompt_tokens.copy(),
                "n_predict": generation_tokens,
                "trial_index": trial_index,
                "ignore_eos": ignore_eos,
            }
        )
        if self.engine.on_complete:
            self.engine.on_complete(len(self.engine.requests), cancel)
        events = [
            {
                "arrival_ns": (20 + index * 30) * 1_000_000,
                "tokens": [100 + index],
                "content": "test",
            }
            for index in range(generation_tokens)
        ]
        return {
            "trial_index": trial_index,
            "request_latency_ms": 150.0,
            "ttft_ms": 20.0,
            "prompt_tokens": len(prompt_tokens),
            "prompt_processed_tokens": len(prompt_tokens),
            "generated_tokens": generation_tokens,
            "target_generated_tokens": generation_tokens,
            "prefill_tokens_per_second": 4000.0,
            "decode_tokens_per_second": 100.0 / 3,
            "mean_inter_token_latency_ms": 30.0,
            "median_inter_token_latency_ms": 30.0,
            "cache_reused_tokens": 0,
            "early_stop": False,
            "finish_reason": "limit",
            "output": "TEST ONLY deterministic generated content",
            "events": events,
            "request": {
                "prompt": prompt_tokens.copy(),
                "n_predict": generation_tokens,
                "cache_prompt": False,
                "ignore_eos": ignore_eos,
            },
            "response": {"TEST_ONLY": True, "stop": True},
            "warnings": [],
        }


class TestRequestEngine:
    __test__ = False

    def __init__(
        self,
        fixture_text: str,
        *,
        fixture_token_count: int = 64,
        on_probe: Callable[[], None] | None = None,
        on_complete: Callable[[int, threading.Event], None] | None = None,
    ) -> None:
        self.fixture_text = fixture_text
        self.fixture_token_count = fixture_token_count
        self.on_probe = on_probe
        self.on_complete = on_complete
        self.enters = 0
        self.exits = 0
        self.tokenize_calls = 0
        self.requests: list[dict[str, Any]] = []
        self.sessions: list[TestRequestSession] = []

    def probe(self) -> dict[str, Any]:
        if self.on_probe:
            self.on_probe()
        return {"TEST_ONLY": True, "version": "in-process test backend"}

    def session(self, settings: Any, cancel: threading.Event) -> TestRequestSession:
        session = TestRequestSession(self)
        self.sessions.append(session)
        return session


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RequestConfig:
    model = tmp_path / "TEST_ONLY.gguf"
    model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("TEST ONLY text fixture used by an in-process test backend.\n")
    monkeypatch.setattr(request_runner, "environment", lambda: {"TEST_ONLY": True})
    monkeypatch.setattr(request_runner, "benchmark_lock", nullcontext)
    return RequestConfig(
        name="TEST ONLY warm request orchestration",
        model=model,
        binary=tmp_path / "unused-in-process-test-engine",
        fixtures=(fixture,),
        prompt_tokens=(8,),
        generation_tokens=(4,),
        threads=(1,),
        gpu_layers=(0,),
        batch_sizes=(32,),
        ubatch_sizes=(16,),
        context_size=64,
    )


def experiment(config: RequestConfig, engine: TestRequestEngine, store: Store) -> dict[str, Any]:
    return store.get_experiment(request_runner.run_request_experiment(config, engine, store))


def test_warmup_is_separate_and_actual_token_prefix_is_preserved(
    config: RequestConfig, tmp_path: Path
) -> None:
    engine = TestRequestEngine(config.fixtures[0].read_text())
    store = Store(tmp_path / "results.sqlite3")
    result = experiment(config, engine, store)
    assert result["status"] == "completed"
    assert len(engine.requests) == 6
    assert engine.enters == engine.exits == 1
    assert engine.tokenize_calls == 1
    case = result["cases"][0]
    assert case["samples"] == []
    assert len(case["request_samples"]) == 5
    assert [sample["trial_index"] for sample in case["request_samples"]] == list(range(5))
    assert all(sample["request_latency_ms"] == 150.0 for sample in case["request_samples"])
    assert all(sample["ttft_ms"] == 20.0 for sample in case["request_samples"])
    assert len(case["process"]["warmup_request_samples"]) == 1
    assert case["process"]["tokenizer"] == {
        "add_special": True,
        "parse_special": False,
        "fixture_token_count": 64,
        "prompt_token_ids": list(range(8)),
    }
    assert all(request["prompt"] == list(range(8)) for request in engine.requests)
    assert case["process"]["server"]["startup_duration_ms"] == 2000.0
    assert case["process"]["server"]["closed"] is True
    assert case["process"]["server"]["returncode"] == 0
    assert "final child log" in case["process"]["server"]["stderr"]
    assert result["metadata"]["mode"] == "request_warm"


def test_short_fixture_fails_without_request_samples_and_closes_session(
    config: RequestConfig, tmp_path: Path
) -> None:
    engine = TestRequestEngine(config.fixtures[0].read_text(), fixture_token_count=3)
    result = experiment(config, engine, Store(tmp_path / "results.sqlite3"))
    case = result["cases"][0]
    assert result["status"] == case["status"] == "failed"
    assert "Fixture has 3 model tokens; requested 8" in case["error"]
    assert case["request_samples"] == []
    assert engine.requests == []
    assert engine.enters == engine.exits == 1


def test_model_change_after_fingerprinting_prevents_server_start(
    config: RequestConfig, tmp_path: Path
) -> None:
    def mutate() -> None:
        config.model.write_bytes(config.model.read_bytes() + b"TEST ONLY change")

    engine = TestRequestEngine(config.fixtures[0].read_text(), on_probe=mutate)
    result = experiment(config, engine, Store(tmp_path / "results.sqlite3"))
    assert result["status"] == "failed"
    case = result["cases"][0]
    assert "Model changed after fingerprinting" in case["error"]
    assert case["request_samples"] == []
    assert engine.enters == engine.exits == 0
    assert engine.requests == []


def test_model_change_during_measurement_rejects_timings_and_keeps_raw_evidence(
    config: RequestConfig, tmp_path: Path
) -> None:
    def mutate(call: int, _: threading.Event) -> None:
        if call == 3:  # one warmup, one valid measurement, then changed model
            config.model.write_bytes(config.model.read_bytes() + b"TEST ONLY during request")

    engine = TestRequestEngine(config.fixtures[0].read_text(), on_complete=mutate)
    result = experiment(config, engine, Store(tmp_path / "results.sqlite3"))
    case = result["cases"][0]
    assert case["status"] == "failed"
    assert "Model changed after fingerprinting" in case["error"]
    assert case["request_samples"] == []
    assert len(case["process"]["partial_request_samples"]) == 1
    assert case["process"]["last_request_sample"]["output"].startswith("TEST ONLY")
    assert engine.enters == engine.exits == 1


def test_timeout_keeps_partial_samples_out_of_accepted_aggregates(
    config: RequestConfig, tmp_path: Path
) -> None:
    def timeout(call: int, _: threading.Event) -> None:
        if call == 3:
            raise TimeoutError("TEST ONLY request deadline expired")

    engine = TestRequestEngine(config.fixtures[0].read_text(), on_complete=timeout)
    result = experiment(config, engine, Store(tmp_path / "results.sqlite3"))
    case = result["cases"][0]
    assert result["status"] == "failed"
    assert case["status"] == "timeout"
    assert case["request_samples"] == []
    assert len(case["process"]["partial_request_samples"]) == 1
    assert engine.enters == engine.exits == 1


def test_cancellation_marks_current_and_queued_cases_and_restores_signals(
    config: RequestConfig, tmp_path: Path
) -> None:
    config = replace(config, prompt_tokens=(8, 16))

    def stop(_: int, cancel: threading.Event) -> None:
        cancel.set()
        raise InterruptedError("TEST ONLY canceled during warmup")

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    engine = TestRequestEngine(config.fixtures[0].read_text(), on_complete=stop)
    result = experiment(config, engine, Store(tmp_path / "results.sqlite3"))
    assert result["status"] == "canceled"
    assert len(result["cases"]) == 2
    assert all(case["status"] == "canceled" for case in result["cases"])
    assert all(case["request_samples"] == [] for case in result["cases"])
    assert len(engine.requests) == 1
    assert engine.enters == engine.exits == 1
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_case_insertion_failure_does_not_leave_running_experiment(
    config: RequestConfig, tmp_path: Path
) -> None:
    class FailingStore(Store):
        def add_case(self, experiment_id: str, position: int, config: dict[str, Any]) -> str:
            if position == 1:
                raise RuntimeError("TEST ONLY insertion failure")
            return super().add_case(experiment_id, position, config)

    engine = TestRequestEngine(config.fixtures[0].read_text())
    store = FailingStore(tmp_path / "results.sqlite3")
    with pytest.raises(RuntimeError, match="insertion failure"):
        request_runner.run_request_experiment(replace(config, prompt_tokens=(8, 16)), engine, store)
    result = store.get_experiment(store.list_experiments()[0]["id"])
    assert result["status"] == "failed"
    assert result["cases"][0]["status"] == "canceled"
    assert engine.enters == engine.exits == 0


def test_canceled_preflight_creates_no_experiment_and_restores_signals(
    config: RequestConfig, tmp_path: Path
) -> None:
    store = Store(tmp_path / "results.sqlite3")
    engine = TestRequestEngine(config.fixtures[0].read_text())
    cancel = threading.Event()
    cancel.set()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(InterruptedError, match="Canceled"):
        request_runner.run_request_experiment(config, engine, store, cancel)
    assert store.list_experiments() == []
    assert engine.enters == engine.exits == 0
    assert {sig: signal.getsignal(sig) for sig in previous} == previous
