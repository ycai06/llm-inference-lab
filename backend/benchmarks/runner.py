"""One local worker with persistent outcomes, including interrupted cases."""

from __future__ import annotations

import signal
import sys
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from backend.benchmarks.environment import environment
from backend.benchmarks.locking import benchmark_lock
from backend.benchmarks.model import inspect_model
from backend.engines.base import BenchmarkEngine
from backend.engines.llama_bench import BenchCase
from backend.schemas.config import ExperimentConfig
from backend.storage.database import Store


def now() -> str:
    return datetime.now(UTC).isoformat()


def run_experiment(
    config: ExperimentConfig,
    engine: BenchmarkEngine,
    store: Store,
    cancel: threading.Event | None = None,
) -> str:
    cancel = cancel if cancel is not None else threading.Event()
    previous_handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda _sig, _frame: cancel.set())
    try:
        with benchmark_lock():
            print("Fingerprinting model and inspecting engine…", file=sys.stderr, flush=True)
            model = inspect_model(config.model, cancel)
            probe = engine.probe()
            if cancel.is_set():
                raise InterruptedError("Canceled during preflight")
            metadata = {
                "schema_version": 1,
                "environment": environment(),
                "model": model,
                "engine": probe,
                "workload": "upstream synthetic vocabulary token IDs; no natural-text fixture",
                "timing_source": "llama-bench samples_ns (model loading excluded)",
                "warmup_policy": "upstream native warmup per case; prompt + one decode token",
                "kv_policy": "upstream clears KV memory before each measured repetition",
                "sampling": "not applicable: synthetic token IDs, no sampled generation",
                "order": "seeded random case order; repetitions contiguous within each case",
                "unavailable_metrics": [
                    "ttft_ms",
                    "request_latency_ms",
                    "inter_token_latency_ms",
                    "gpu_memory_bytes",
                    "gpu_utilization",
                    "power_watts",
                ],
                "warnings": (
                    ["Fewer than five measured repetitions"] if config.repetitions < 5 else []
                ),
            }
            experiment_id = store.create_experiment(config.name, config.to_dict(), metadata)
            failures = 0
            try:
                planned = [
                    (store.add_case(experiment_id, i, c), c) for i, c in enumerate(config.cases())
                ]
                for position, (case_id, case_config) in enumerate(planned):
                    if cancel.is_set():
                        store.finish_case(case_id, "canceled", {}, [], "Canceled before execution")
                        continue
                    case = BenchCase(**case_config)
                    print(f"[{position + 1}/{len(planned)}] {case}", file=sys.stderr, flush=True)
                    process: dict[str, Any] = {"started_at": now()}
                    try:
                        current = config.model.stat()
                        if (current.st_size, current.st_mtime_ns, current.st_ino) != (
                            model["size_bytes"],
                            model["mtime_ns"],
                            model["inode"],
                        ):
                            raise ValueError("Model changed after fingerprinting")
                        outcome = engine.run(
                            config.model, case, config.repetitions, config.timeout_s, cancel
                        )
                        process.update(asdict(outcome), finished_at=now())
                        current = config.model.stat()
                        if (current.st_size, current.st_mtime_ns, current.st_ino) != (
                            model["size_bytes"],
                            model["mtime_ns"],
                            model["inode"],
                        ):
                            raise ValueError("Model changed during inference; samples rejected")
                        samples, raw = (
                            engine.parse(outcome.stdout, case, config.repetitions)
                            if outcome.status == "completed"
                            else ([], {})
                        )
                        process["upstream"] = raw
                        error = (
                            None
                            if outcome.status == "completed"
                            else (
                                f"Engine {outcome.status} (exit {outcome.returncode}); "
                                "see saved stderr"
                            )
                        )
                        store.finish_case(case_id, outcome.status, process, samples, error)
                        if outcome.status == "canceled":
                            cancel.set()
                        elif outcome.status != "completed":
                            failures += 1
                    except (ValueError, OSError, RuntimeError) as exc:
                        failures += 1
                        process["finished_at"] = now()
                        store.finish_case(case_id, "failed", process, [], str(exc))
                status = "canceled" if cancel.is_set() else "failed" if failures else "completed"
                store.finish_experiment(
                    experiment_id, status, f"{failures} case(s) failed" if failures else None
                )
            except BaseException as exc:
                # Even unexpected errors must not leave a running experiment in the store.
                snapshot = store.get_experiment(experiment_id)
                for case_row in snapshot["cases"]:
                    if case_row["status"] == "queued":
                        store.finish_case(case_row["id"], "canceled", {}, [], "Worker interrupted")
                store.finish_experiment(experiment_id, "failed", str(exc))
                raise
            return experiment_id
    finally:
        for previous_sig, handler in previous_handlers.items():
            signal.signal(previous_sig, handler)
