"""Orchestrator contracts using an in-process TEST ONLY backend and tiny GGUF header."""

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

from backend.benchmarks import runner
from backend.engines.base import BenchmarkEngine
from backend.engines.llama_bench import BenchCase
from backend.engines.process import ProcessResult
from backend.schemas.config import ExperimentConfig
from backend.storage.database import Store


class DeterministicTestEngine:
    """TEST ONLY Protocol implementation: never executes a subprocess or loads weights."""

    def __init__(
        self,
        *,
        on_probe: Callable[[], None] | None = None,
        on_run: Callable[[Path], None] | None = None,
    ) -> None:
        self.on_probe = on_probe
        self.on_run = on_run
        self.probe_calls = 0
        self.run_calls = 0
        self.parse_calls = 0

    def probe(self) -> dict[str, Any]:
        self.probe_calls += 1
        if self.on_probe is not None:
            self.on_probe()
        return {"TEST_ONLY": True, "version": "in-process-unit-fixture"}

    def run(
        self,
        model: Path,
        case: BenchCase,
        repetitions: int,
        timeout_s: float,
        cancel: threading.Event | None = None,
    ) -> ProcessResult:
        self.run_calls += 1
        if self.on_run is not None:
            self.on_run(model)
        return ProcessResult(
            argv=["TEST-ONLY-NO-PROCESS"],
            stdout="TEST ONLY deterministic backend output",
            stderr="TEST ONLY no inference was performed",
            returncode=0,
            wall_ns=100,
            status="completed",
            sampled_peak_rss_bytes=None,
            sampled_cpu_seconds=None,
            resource_samples=0,
        )

    def parse(
        self,
        stdout: str,
        case: BenchCase,
        repetitions: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.parse_calls += 1
        assert stdout == "TEST ONLY deterministic backend output"
        return [
            {
                "trial_index": trial,
                "duration_ns": 1_000_000_000 + trial * 1_000_000,
                "tokens": case.tokens,
                "tokens_per_second": case.tokens * 1e9 / (1_000_000_000 + trial * 1_000_000),
            }
            for trial in range(repetitions)
        ], {"TEST_ONLY": True}


@pytest.fixture(autouse=True)
def isolate_environment_and_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    # These unit tests must not probe hardware or acquire the real benchmark lock.
    monkeypatch.setattr(runner, "environment", lambda: {"TEST_ONLY": True})
    monkeypatch.setattr(runner, "benchmark_lock", nullcontext)


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    model = tmp_path / "test-only-header.gguf"
    model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    return ExperimentConfig(
        name="TEST ONLY runner contract",
        model=model,
        binary=tmp_path / "unused-test-only-binary",
        prompt_tokens=(8,),
        generation_tokens=(4,),
        threads=(1,),
        batch_sizes=(32,),
        ubatch_sizes=(16,),
        repetitions=5,
    )


def test_engine_protocol_can_be_injected_without_llama_or_a_subprocess(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    engine = DeterministicTestEngine()
    injected: BenchmarkEngine = engine
    store = Store(tmp_path / "results.sqlite3")
    experiment_id = runner.run_experiment(config, injected, store)
    experiment = store.get_experiment(experiment_id)
    assert experiment["status"] == "completed"
    assert engine.probe_calls == 1
    assert engine.run_calls == 2
    assert engine.parse_calls == 2
    assert experiment["metadata"]["engine"]["TEST_ONLY"] is True
    assert experiment["metadata"]["environment"] == {"TEST_ONLY": True}
    for case in experiment["cases"]:
        assert case["status"] == "completed"
        assert len(case["samples"]) == 5
        assert case["process"]["upstream"] == {"TEST_ONLY": True}
        assert case["process"]["stdout"] == "TEST ONLY deterministic backend output"
        assert case["process"]["sampled_peak_rss_bytes"] is None
        assert [sample["trial_index"] for sample in case["samples"]] == list(range(5))


def test_case_insertion_failure_marks_experiment_and_inserted_cases_terminal(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    class FailingCaseStore(Store):
        def add_case(
            self,
            experiment_id: str,
            position: int,
            config: dict[str, Any],
        ) -> str:
            if position == 1:
                raise RuntimeError("TEST ONLY case insertion failure")
            return super().add_case(experiment_id, position, config)

    store = FailingCaseStore(tmp_path / "results.sqlite3")
    engine = DeterministicTestEngine()
    with pytest.raises(RuntimeError, match="case insertion failure"):
        runner.run_experiment(config, engine, store)
    experiments = store.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["status"] == "failed"
    assert experiments[0]["finished_at"] is not None
    experiment = store.get_experiment(experiments[0]["id"])
    assert len(experiment["cases"]) == 1
    assert experiment["cases"][0]["status"] == "canceled"
    assert experiment["cases"][0]["samples"] == []
    assert engine.run_calls == 0


def test_model_change_after_fingerprinting_prevents_engine_execution(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    config = replace(config, generation_tokens=())

    def change_test_file() -> None:
        config.model.write_bytes(config.model.read_bytes() + b"TEST ONLY change")

    engine = DeterministicTestEngine(on_probe=change_test_file)
    store = Store(tmp_path / "results.sqlite3")
    experiment = store.get_experiment(runner.run_experiment(config, engine, store))
    assert experiment["status"] == "failed"
    case = experiment["cases"][0]
    assert case["status"] == "failed"
    assert "Model changed after fingerprinting" in case["error"]
    assert case["samples"] == []
    assert engine.run_calls == 0
    assert engine.parse_calls == 0


def test_model_change_during_execution_rejects_samples_but_retains_output(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    config = replace(config, generation_tokens=())

    def change_test_file(model: Path) -> None:
        model.write_bytes(model.read_bytes() + b"TEST ONLY during-run change")

    engine = DeterministicTestEngine(on_run=change_test_file)
    store = Store(tmp_path / "results.sqlite3")
    experiment = store.get_experiment(runner.run_experiment(config, engine, store))
    assert experiment["status"] == "failed"
    case = experiment["cases"][0]
    assert case["status"] == "failed"
    assert "Model changed during inference; samples rejected" in case["error"]
    assert case["samples"] == []
    assert case["process"]["stdout"] == "TEST ONLY deterministic backend output"
    assert engine.run_calls == 1
    assert engine.parse_calls == 0


def test_pre_canceled_preflight_creates_no_experiment_and_restores_signal_handlers(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    engine = DeterministicTestEngine()
    store = Store(tmp_path / "results.sqlite3")
    cancel = threading.Event()
    cancel.set()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(InterruptedError, match="Canceled"):
        runner.run_experiment(config, engine, store, cancel)
    assert store.list_experiments() == []
    assert engine.probe_calls == 0
    assert engine.run_calls == 0
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_cancellation_during_probe_creates_no_experiment(
    config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    cancel = threading.Event()
    engine = DeterministicTestEngine(on_probe=cancel.set)
    store = Store(tmp_path / "results.sqlite3")
    with pytest.raises(InterruptedError, match="Canceled during preflight"):
        runner.run_experiment(config, engine, store, cancel)
    assert store.list_experiments() == []
    assert engine.probe_calls == 1
    assert engine.run_calls == 0
