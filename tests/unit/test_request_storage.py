"""Request persistence contracts use visibly isolated, synthetic test observations."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from backend.storage.database import Store


def request_sample(**overrides: Any) -> dict[str, Any]:
    return {
        "trial_index": 0,
        "request_latency_ms": 400.0,
        "ttft_ms": 100.0,
        "prompt_tokens": 128,
        "prompt_processed_tokens": 128,
        "generated_tokens": 8,
        "target_generated_tokens": 8,
        "prefill_tokens_per_second": 1280.0,
        "decode_tokens_per_second": 23.0,
        "mean_inter_token_latency_ms": None,
        "median_inter_token_latency_ms": None,
        "cache_reused_tokens": 0,
        "early_stop": False,
        "finish_reason": "limit",
        "output": "TEST ONLY generated text",
        "events": [{"arrival_ns": 100_000_000, "tokens": [42], "content": "TEST ONLY"}],
        "request": {"prompt": [1, 2], "stream": True, "cache_prompt": False},
        "response": {"stop": True, "tokens_predicted": 8, "TEST_ONLY": True},
        "warnings": ["TEST ONLY incomplete token event coverage"],
    } | overrides


def setup_case(tmp_path: Path) -> tuple[Store, str, str]:
    store = Store(tmp_path / "requests.sqlite3")
    experiment_id = store.create_experiment(
        "TEST ONLY requests",
        {"mode": "request_warm", "repetitions": 5},
        {"TEST_ONLY": True, "model_sha256": "fixture-hash"},
    )
    case_id = store.add_case(
        experiment_id,
        0,
        {
            "kind": "request",
            "prompt_tokens": 128,
            "generation_tokens": 8,
            "threads": 4,
            "gpu_layers": -1,
            "batch_size": 512,
            "ubatch_size": 128,
            "context_size": 1024,
        },
    )
    return store, experiment_id, case_id


def test_request_roundtrip_exports_and_metadata_update(tmp_path: Path) -> None:
    store, experiment_id, case_id = setup_case(tmp_path)
    sample = request_sample()
    process = {"argv": ["TEST ONLY server"], "stderr": "test log"}
    store.finish_request_case(case_id, "completed", process, [sample])
    metadata = {"TEST_ONLY": True, "server": {"shutdown": "clean"}}
    store.update_metadata(experiment_id, metadata)
    store.finish_experiment(experiment_id, "completed")
    result = Store(store.path).get_experiment(experiment_id)
    assert result["metadata"] == metadata
    assert result["cases"][0]["request_samples"] == [sample]
    assert result["cases"][0]["samples"] == []
    assert result["cases"][0]["process"] == process

    store.export_json(experiment_id, tmp_path / "request.json")
    assert json.loads((tmp_path / "request.json").read_text()) == result
    store.export_csv(experiment_id, tmp_path / "request.csv")
    with (tmp_path / "request.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "request_warm"
    assert row["ttft_ms"] == "100.0"
    assert row["generated_tokens"] == "8"
    assert row["requested_prompt_tokens"] == "128"
    assert row["mean_inter_token_latency_ms"] == ""
    assert json.loads(row["sample_json"]) == sample
    assert json.loads(row["events_json"]) == sample["events"]
    assert json.loads(row["metadata_json"]) == metadata
    assert "duration_ns" not in row


def test_migration_from_v1_preserves_synthetic_data(tmp_path: Path) -> None:
    path = tmp_path / "v1.sqlite3"
    initial = Path(__file__).parents[2] / "backend/storage/migrations/001_initial.sql"
    with sqlite3.connect(path) as connection:
        connection.executescript(initial.read_text())
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("old", "TEST ONLY v1", "completed", "2026-01-01", "2026-01-01", "{}", "{}", None),
        )
        connection.execute(
            "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("case", "old", 0, "{}", "completed", "{}", None),
        )
        connection.execute("INSERT INTO samples VALUES ('case', 0, 1000000, 2, 2000.0)")
    store = Store(path)
    migrated = store.get_experiment("old")["cases"][0]
    assert migrated["samples"] == [
        {
            "trial_index": 0,
            "duration_ns": 1_000_000,
            "tokens": 2,
            "tokens_per_second": 2000.0,
        }
    ]
    assert migrated["request_samples"] == []
    assert Store(path).get_experiment("old") == store.get_experiment("old")
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


@pytest.mark.parametrize("status", ["failed", "timeout", "canceled"])
def test_failed_request_case_exports_blank_measurements(tmp_path: Path, status: str) -> None:
    store, experiment_id, case_id = setup_case(tmp_path)
    store.finish_request_case(case_id, status, {"stderr": "TEST ONLY failure"}, [], "test failure")
    store.export_csv(experiment_id, tmp_path / "failure.csv")
    with (tmp_path / "failure.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["case_status"] == status
    assert rows[0]["ttft_ms"] == ""
    assert rows[0]["error"] == "test failure"
    assert store.get_experiment(experiment_id)["cases"][0]["request_samples"] == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"trial_index": True},
        {"prompt_tokens": 0},
        {"generated_tokens": -1},
        {"target_generated_tokens": 0},
        {"cache_reused_tokens": 1.5},
        {"request_latency_ms": 0},
        {"request_latency_ms": float("nan")},
        {"ttft_ms": float("inf")},
        {"ttft_ms": -1},
        {"ttft_ms": 401},
        {"prefill_tokens_per_second": "fast"},
        {"decode_tokens_per_second": 0},
        {"mean_inter_token_latency_ms": True},
        {"median_inter_token_latency_ms": -1},
        {"generated_tokens": 9},
        {"early_stop": True},
        {"early_stop": 0},
        {"warnings": "warning"},
        {"warnings": [42]},
        {"events": ["event"]},
        {"response": []},
        {"request": {"invalid": float("nan")}},
        {"output": None},
    ],
)
def test_invalid_request_is_rejected_atomically(tmp_path: Path, overrides: dict[str, Any]) -> None:
    store, experiment_id, case_id = setup_case(tmp_path)
    with pytest.raises(ValueError):
        store.finish_request_case(case_id, "completed", {}, [request_sample(**overrides)])
    case = store.get_experiment(experiment_id)["cases"][0]
    assert case["status"] == "queued"
    assert case["process"] is None
    assert case["request_samples"] == []


def test_zero_generated_tokens_and_missing_optional_metrics_are_valid(tmp_path: Path) -> None:
    store, experiment_id, case_id = setup_case(tmp_path)
    sample = request_sample(
        generated_tokens=0,
        early_stop=True,
        ttft_ms=None,
        prefill_tokens_per_second=None,
        decode_tokens_per_second=None,
        events=[],
        output="",
        finish_reason="eos",
    )
    store.finish_request_case(case_id, "completed", {}, [sample])
    assert store.get_experiment(experiment_id)["cases"][0]["request_samples"] == [sample]


def test_missing_required_field_rejected_and_duplicate_trials_roll_back(tmp_path: Path) -> None:
    store, experiment_id, case_id = setup_case(tmp_path)
    sample = request_sample()
    incomplete = dict(sample)
    del incomplete["ttft_ms"]
    with pytest.raises(ValueError, match="missing required"):
        store.finish_request_case(case_id, "completed", {}, [incomplete])
    with pytest.raises(sqlite3.IntegrityError):
        store.finish_request_case(case_id, "completed", {}, [sample, sample])
    case = store.get_experiment(experiment_id)["cases"][0]
    assert case["status"] == "queued"
    assert case["request_samples"] == []


def test_case_cannot_mix_request_and_synthetic_samples(tmp_path: Path) -> None:
    store, experiment_id, case_id = setup_case(tmp_path)
    store.finish_request_case(case_id, "completed", {}, [request_sample()])
    synthetic = {"trial_index": 0, "duration_ns": 1000, "tokens": 2, "tokens_per_second": 2e6}
    with pytest.raises(ValueError, match="cannot mix"):
        store.finish_case(case_id, "completed", {}, [synthetic])
    other = store.add_case(experiment_id, 1, {})
    store.finish_case(other, "completed", {}, [synthetic])
    with pytest.raises(ValueError, match="cannot mix"):
        store.finish_request_case(other, "completed", {}, [request_sample()])


def test_request_status_and_unknown_id_validation(tmp_path: Path) -> None:
    store, _, case_id = setup_case(tmp_path)
    with pytest.raises(ValueError, match="terminal case status"):
        store.finish_request_case(case_id, "unknown", {}, [])
    with pytest.raises(ValueError, match="cannot contain valid"):
        store.finish_request_case(case_id, "failed", {}, [request_sample()])
    with pytest.raises(KeyError, match="Unknown case"):
        store.finish_request_case("missing", "failed", {}, [])
    with pytest.raises(KeyError, match="Unknown experiment"):
        store.update_metadata("missing", {})
