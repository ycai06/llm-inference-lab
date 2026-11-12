"""Small, versioned SQLite store; raw metadata remains available in JSON exports."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 1
EXPERIMENT_TERMINAL = {"completed", "failed", "canceled"}
CASE_TERMINAL = {"completed", "failed", "timeout", "canceled"}


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    """Open one connection per operation so API/worker threads do not share handles."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise ValueError(
                    f"Database schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            for next_version in range(version + 1, SCHEMA_VERSION + 1):
                migration = next(
                    (Path(__file__).parent / "migrations").glob(f"{next_version:03d}_*.sql")
                )
                # executescript otherwise commits before executing. A script-local transaction
                # makes schema changes and the version marker succeed or roll back together.
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + migration.read_text(encoding="utf-8")
                    + f"\nPRAGMA user_version = {next_version};\nCOMMIT;"
                )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_experiment(self, name: str, config: dict[str, Any], metadata: dict[str, Any]) -> str:
        experiment_id = str(uuid4())
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO experiments "
                "(id, name, status, created_at, config_json, metadata_json) "
                "VALUES (?, ?, 'running', ?, ?, ?)",
                (experiment_id, name, _now(), _json(config), _json(metadata)),
            )
        return experiment_id

    def add_case(self, experiment_id: str, position: int, config: dict[str, Any]) -> str:
        case_id = str(uuid4())
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO cases (id, experiment_id, position, config_json, status) "
                "VALUES (?, ?, ?, ?, 'queued')",
                (case_id, experiment_id, position, _json(config)),
            )
        return case_id

    def finish_case(
        self,
        case_id: str,
        status: str,
        process: dict[str, Any],
        samples: list[dict[str, Any]],
        error: str | None = None,
    ) -> None:
        if status not in CASE_TERMINAL:
            raise ValueError(f"Invalid terminal case status: {status}")
        if samples and status != "completed":
            raise ValueError("Failed, timed-out, or canceled cases cannot contain valid samples")
        process_json = _json(process)
        normalized = []
        for sample in samples:
            throughput = sample["tokens_per_second"]
            if (
                isinstance(throughput, bool)
                or not isinstance(throughput, (int, float))
                or not math.isfinite(throughput)
                or throughput <= 0
            ):
                raise ValueError("Sample throughput must be positive and finite")
            for key in ("trial_index", "duration_ns", "tokens"):
                if isinstance(sample[key], bool) or not isinstance(sample[key], int):
                    raise ValueError(f"{key} must be an integer")
            normalized.append(
                (
                    case_id,
                    sample["trial_index"],
                    sample["duration_ns"],
                    sample["tokens"],
                    throughput,
                )
            )
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE cases SET status = ?, process_json = ?, error = ? WHERE id = ?",
                (status, process_json, error, case_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown case: {case_id}")
            connection.execute("DELETE FROM samples WHERE case_id = ?", (case_id,))
            connection.executemany(
                "INSERT INTO samples "
                "(case_id, trial_index, duration_ns, tokens, tokens_per_second) "
                "VALUES (?, ?, ?, ?, ?)",
                normalized,
            )

    def finish_experiment(self, experiment_id: str, status: str, error: str | None = None) -> None:
        if status not in EXPERIMENT_TERMINAL:
            raise ValueError(f"Invalid terminal experiment status: {status}")
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE experiments SET status = ?, finished_at = ?, error = ? WHERE id = ?",
                (status, _now(), error, experiment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown experiment: {experiment_id}")

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown experiment: {experiment_id}")
            experiment = dict(row)
            experiment["config"] = json.loads(experiment.pop("config_json"))
            experiment["metadata"] = json.loads(experiment.pop("metadata_json"))
            cases = []
            for case_row in connection.execute(
                "SELECT * FROM cases WHERE experiment_id = ? ORDER BY position",
                (experiment_id,),
            ):
                case = dict(case_row)
                case["config"] = json.loads(case.pop("config_json"))
                process_json = case.pop("process_json")
                case["process"] = json.loads(process_json) if process_json else None
                case["samples"] = [
                    dict(sample)
                    for sample in connection.execute(
                        "SELECT trial_index, duration_ns, tokens, tokens_per_second "
                        "FROM samples WHERE case_id = ? ORDER BY trial_index",
                        (case["id"],),
                    )
                ]
                cases.append(case)
            experiment["cases"] = cases
            return experiment

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT e.id, e.name, e.status, e.created_at, e.finished_at, e.error, "
                    "(SELECT COUNT(*) FROM cases c WHERE c.experiment_id = e.id) AS case_count "
                    "FROM experiments e ORDER BY e.created_at DESC, e.id"
                )
            ]

    def export_json(self, experiment_id: str, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.get_experiment(experiment_id), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def export_csv(self, experiment_id: str, path: Path) -> None:
        experiment = self.get_experiment(experiment_id)
        fields = [
            "experiment_id",
            "experiment_name",
            "experiment_status",
            "created_at",
            "case_id",
            "case_position",
            "case_status",
            "kind",
            "tokens",
            "threads",
            "gpu_layers",
            "batch_size",
            "ubatch_size",
            "flash_attention",
            "trial_index",
            "duration_ns",
            "duration_ms",
            "tokens_per_second",
            "error",
            "case_config_json",
            "process_json",
            "experiment_config_json",
            "metadata_json",
        ]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for case in experiment["cases"]:
                base = {
                    "experiment_id": experiment["id"],
                    "experiment_name": experiment["name"],
                    "experiment_status": experiment["status"],
                    "created_at": experiment["created_at"],
                    "case_id": case["id"],
                    "case_position": case["position"],
                    "case_status": case["status"],
                    "error": case["error"],
                    "case_config_json": _json(case["config"]),
                    "process_json": _json(case["process"]),
                    "experiment_config_json": _json(experiment["config"]),
                    "metadata_json": _json(experiment["metadata"]),
                }
                for key in (
                    "kind",
                    "tokens",
                    "threads",
                    "gpu_layers",
                    "batch_size",
                    "ubatch_size",
                    "flash_attention",
                ):
                    base[key] = case["config"].get(key)
                for sample in case["samples"] or [{}]:
                    row = base | sample
                    if "duration_ns" in sample:
                        row["duration_ms"] = sample["duration_ns"] / 1_000_000
                    writer.writerow(row)
