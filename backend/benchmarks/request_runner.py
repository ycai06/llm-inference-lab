"""Warm native-completion requests with exact fixture token IDs and separate timing."""

from __future__ import annotations

import hashlib
import signal
import sys
import threading
from datetime import UTC, datetime
from typing import Any

from backend.benchmarks.environment import environment
from backend.benchmarks.locking import benchmark_lock
from backend.benchmarks.model import inspect_model
from backend.engines.llama_server import ServerSettings
from backend.engines.request_base import RequestEngine, RequestSession
from backend.schemas.request_config import MAX_FIXTURE_BYTES, RequestConfig
from backend.storage.database import Store


def _check_model(config: RequestConfig, model: dict[str, Any]) -> None:
    current = config.model.stat()
    if (current.st_size, current.st_mtime_ns, current.st_ino) != (
        model["size_bytes"],
        model["mtime_ns"],
        model["inode"],
    ):
        raise ValueError("Model changed after fingerprinting; request samples rejected")


def run_request_experiment(
    config: RequestConfig,
    engine: RequestEngine,
    store: Store,
    cancel: threading.Event | None = None,
) -> str:
    cancel = cancel if cancel is not None else threading.Event()
    handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda _sig, _frame: cancel.set())
    try:
        with benchmark_lock():
            print("Fingerprinting model and inspecting server…", file=sys.stderr, flush=True)
            model = inspect_model(config.model, cancel)
            probe = engine.probe()
            if cancel.is_set():
                raise InterruptedError("Canceled during server preflight")
            fixtures = {}
            for path in config.fixtures:
                with path.open("rb") as source:
                    raw = source.read(MAX_FIXTURE_BYTES + 1)
                if len(raw) > MAX_FIXTURE_BYTES:
                    raise ValueError("Fixture exceeds size limit at execution time")
                fixtures[str(path)] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "source_text": raw.decode("utf-8"),
                }
            metadata = {
                "schema_version": 2,
                "environment": environment(),
                "model": model,
                "engine": probe,
                "mode": "request_warm",
                "workload": "native completion over an exact prefix of tokenized text fixture",
                "timing_source": "client monotonic recv-chunk timestamps and native server timings",
                "warmup_policy": f"{config.warmups} full unmeasured request(s) per loaded case",
                "kv_policy": "cache_prompt=false; require timings.cache_n=0 for every request",
                "tokenizer": {
                    "add_special": True,
                    "parse_special": False,
                    "selection": "first N token IDs; sent as IDs without retokenization",
                },
                "sampling": {
                    "seed": config.seed,
                    "temperature": 0,
                    "ignore_eos": config.ignore_eos,
                },
                "order": "seeded randomized cases; contiguous warmup and measured repetitions",
                "unavailable_metrics": [
                    "gpu_memory_bytes",
                    "gpu_utilization",
                    "power_watts",
                    "thermal_pressure",
                ],
                "warnings": ["Within-process repetitions do not capture between-session variance"]
                + (["Fewer than five measured repetitions"] if config.repetitions < 5 else []),
            }
            experiment_id = store.create_experiment(config.name, config.to_dict(), metadata)
            failures = 0
            try:
                planned = [
                    (store.add_case(experiment_id, i, case), case)
                    for i, case in enumerate(config.cases())
                ]
                for position, (case_id, case) in enumerate(planned):
                    if cancel.is_set():
                        store.finish_request_case(
                            case_id, "canceled", {}, [], "Canceled before execution"
                        )
                        continue
                    print(
                        f"[{position + 1}/{len(planned)}] request pp{case['prompt_tokens']} "
                        f"tg{case['generation_tokens']} threads={case['threads']} "
                        f"gpu={case['gpu_layers']}",
                        file=sys.stderr,
                        flush=True,
                    )
                    process: dict[str, Any] = {
                        "started_at": datetime.now(UTC).isoformat(),
                        "fixture": fixtures[case["fixture"]],
                    }
                    settings = ServerSettings(
                        model=config.model,
                        threads=case["threads"],
                        gpu_layers=case["gpu_layers"],
                        batch_size=case["batch_size"],
                        ubatch_size=case["ubatch_size"],
                        context_size=config.context_size,
                        flash_attention=config.flash_attention,
                        startup_timeout_s=config.startup_timeout_s,
                    )
                    session: RequestSession | None = None
                    accepted: list[dict[str, Any]] = []
                    status, error = "completed", None
                    try:
                        _check_model(config, model)
                        session = engine.session(settings, cancel)
                        with session as running:
                            all_tokens = running.tokenize(
                                process["fixture"]["source_text"],
                                add_special=True,
                                parse_special=False,
                                timeout_s=config.timeout_s,
                            )
                            target = case["prompt_tokens"]
                            if len(all_tokens) < target:
                                raise ValueError(
                                    f"Fixture has {len(all_tokens)} model tokens; "
                                    f"requested {target}"
                                )
                            prompt = all_tokens[:target]
                            process["tokenizer"] = {
                                "add_special": True,
                                "parse_special": False,
                                "fixture_token_count": len(all_tokens),
                                "prompt_token_ids": prompt,
                            }
                            _check_model(config, model)
                            for warmup in range(config.warmups):
                                print(
                                    f"  warmup {warmup + 1}/{config.warmups}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                observed = running.complete(
                                    prompt,
                                    case["generation_tokens"],
                                    config.seed,
                                    config.ignore_eos,
                                    False,
                                    config.timeout_s,
                                    cancel,
                                )
                                process.setdefault("warmup_request_samples", []).append(observed)
                            for trial in range(config.repetitions):
                                _check_model(config, model)
                                sample = running.complete(
                                    prompt,
                                    case["generation_tokens"],
                                    config.seed,
                                    config.ignore_eos,
                                    False,
                                    config.timeout_s,
                                    cancel,
                                    trial_index=trial,
                                )
                                process["last_request_sample"] = sample
                                _check_model(config, model)
                                accepted.append(sample)
                                print(
                                    f"  trial {trial + 1}/{config.repetitions}: "
                                    f"{sample['generated_tokens']} output tokens, "
                                    f"{sample['request_latency_ms']:.1f} ms",
                                    file=sys.stderr,
                                    flush=True,
                                )
                    except InterruptedError as exc:
                        cancel.set()
                        status, error = "canceled", str(exc)
                    except TimeoutError as exc:
                        status, error = "timeout", str(exc)
                        failures += 1
                    except Exception as exc:
                        status = "canceled" if cancel.is_set() else "failed"
                        error = f"{type(exc).__name__}: {exc}"
                        if status == "failed":
                            failures += 1
                    finally:
                        process["server"] = (
                            session.metadata if session else {"status": "not_started"}
                        )
                        process["finished_at"] = datetime.now(UTC).isoformat()
                    # A failed case has no accepted aggregate samples. Keep completed trial
                    # evidence separately so an interruption cannot silently erase observations.
                    if status != "completed":
                        process["partial_request_samples"] = accepted
                        accepted = []
                    store.finish_request_case(case_id, status, process, accepted, error)
                terminal = "canceled" if cancel.is_set() else "failed" if failures else "completed"
                store.finish_experiment(
                    experiment_id, terminal, f"{failures} case(s) failed" if failures else None
                )
            except BaseException as exc:
                for case_row in store.get_experiment(experiment_id)["cases"]:
                    if case_row["status"] == "queued":
                        store.finish_request_case(
                            case_row["id"], "canceled", {}, [], "Worker interrupted"
                        )
                store.finish_experiment(experiment_id, "failed", str(exc))
                raise
            return experiment_id
    finally:
        for previous_sig, handler in handlers.items():
            signal.signal(previous_sig, handler)
