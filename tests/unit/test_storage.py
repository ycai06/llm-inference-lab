"""These synthetic rows test persistence only, and never populate production data."""

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from backend.storage.database import Store


def new_experiment(store: Store) -> str:
    return store.create_experiment(
        "test-only synthetic experiment",
        {"repetitions": 5, "model": "test-model.gguf"},
        {"model_sha256": "fixture-only", "environment": {"cpu": "test CPU"}},
    )


def test_schema_migration_roundtrip_and_exports(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "results.sqlite3"
    store = Store(path)
    experiment_id = new_experiment(store)
    case_id = store.add_case(
        experiment_id,
        0,
        {"kind": "decode", "tokens": 2, "threads": 4, "gpu_layers": 0},
    )
    sample = {
        "trial_index": 0,
        "tokens": 2,
        "duration_ns": 20_000_000,
        "tokens_per_second": 100.0,
    }
    process = {"stdout": "raw benchmark output", "stderr": "", "returncode": 0}
    store.finish_case(case_id, "completed", process, [sample])
    store.finish_experiment(experiment_id, "completed")

    reopened = Store(path)
    result = reopened.get_experiment(experiment_id)
    assert result["status"] == "completed"
    assert result["finished_at"]
    assert result["metadata"]["model_sha256"] == "fixture-only"
    assert result["cases"][0]["samples"] == [sample]
    assert result["cases"][0]["process"] == process
    assert reopened.list_experiments()[0]["case_count"] == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1

    json_path, csv_path = tmp_path / "export.json", tmp_path / "export.csv"
    reopened.export_json(experiment_id, json_path)
    reopened.export_csv(experiment_id, csv_path)
    assert json.loads(json_path.read_text()) == result
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["duration_ms"] == "20.0"
    assert rows[0]["tokens_per_second"] == "100.0"
    assert json.loads(rows[0]["metadata_json"]) == result["metadata"]
    assert rows[0]["threads"] == "4"


@pytest.mark.parametrize("status", ["failed", "timeout", "canceled"])
def test_failure_states_are_preserved_in_exports(tmp_path: Path, status: str) -> None:
    store = Store(tmp_path / "db.sqlite3")
    experiment_id = new_experiment(store)
    case_id = store.add_case(experiment_id, 0, {"kind": "prefill", "tokens": 128})
    store.finish_case(case_id, status, {"returncode": -15}, [], "intentional test failure")
    store.finish_experiment(experiment_id, "canceled" if status == "canceled" else "failed")
    result = store.get_experiment(experiment_id)
    assert result["cases"][0]["status"] == status
    assert result["cases"][0]["samples"] == []
    store.export_csv(experiment_id, tmp_path / "failed.csv")
    with (tmp_path / "failed.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["case_status"] == status
    assert rows[0]["error"] == "intentional test failure"
    assert rows[0]["tokens_per_second"] == ""


def test_invalid_sample_rolls_back_case_update(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite3")
    experiment_id = new_experiment(store)
    case_id = store.add_case(experiment_id, 0, {})
    invalid_sample = {"trial_index": 0, "tokens": 2, "duration_ns": 0, "tokens_per_second": 100.0}
    with pytest.raises(sqlite3.IntegrityError):
        store.finish_case(case_id, "completed", {"returncode": 0}, [invalid_sample])
    case = store.get_experiment(experiment_id)["cases"][0]
    assert case["status"] == "queued"
    assert case["process"] is None
    assert case["samples"] == []


def test_reject_nonfinite_sample(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite3")
    experiment_id = new_experiment(store)
    case_id = store.add_case(experiment_id, 0, {})
    with pytest.raises(ValueError, match="positive and finite"):
        store.finish_case(
            case_id,
            "completed",
            {},
            [{"trial_index": 0, "tokens": 2, "duration_ns": 1, "tokens_per_second": float("nan")}],
        )


def test_missing_ids_and_foreign_keys(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite3")
    with pytest.raises(KeyError, match="Unknown experiment"):
        store.get_experiment("missing")
    with pytest.raises(KeyError, match="Unknown experiment"):
        store.finish_experiment("missing", "failed")
    with pytest.raises(KeyError, match="Unknown case"):
        store.finish_case("missing", "failed", {}, [])
    with pytest.raises(sqlite3.IntegrityError):
        store.add_case("missing", 0, {})


def test_reject_future_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")
    with pytest.raises(ValueError, match="newer than supported"):
        Store(path)


def test_empty_database_and_terminal_status_validation(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite3")
    assert store.list_experiments() == []
    experiment_id = new_experiment(store)
    case_id = store.add_case(experiment_id, 0, {})
    with pytest.raises(ValueError, match="terminal case status"):
        store.finish_case(case_id, "unknown", {}, [])
    with pytest.raises(ValueError, match="terminal experiment status"):
        store.finish_experiment(experiment_id, "unknown")
    with pytest.raises(ValueError, match="cannot contain valid samples"):
        store.finish_case(case_id, "failed", {}, [{"trial_index": 0}])
