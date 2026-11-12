"""CLI contracts exercised with explicitly isolated TEST ONLY engine output."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
import yaml

from backend.storage.database import Store

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli_setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    engine = tmp_path / "test-only-llama-bench"
    shutil.copyfile(REPO / "tests" / "fixtures" / "fake_llama_bench.py", engine)
    engine.chmod(0o700)
    model = tmp_path / "test-only-header.gguf"
    model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "TEST ONLY CLI integration",
                "binary": str(engine),
                "model": str(model),
                "prompt_tokens": [8],
                "generation_tokens": [4],
                "threads": [1],
                "gpu_layers": [0],
                "batch_sizes": [32],
                "ubatch_sizes": [16],
                "repetitions": 5,
                "timeout_s": 2,
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path / "results.sqlite3", model


def cli(db: Path, *args: str, behavior: str = "success") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "backend.cli", "--db", str(db), *args],
        cwd=REPO,
        env=os.environ | {"LIOL_TEST_BEHAVIOR": behavior, "TMPDIR": str(db.parent)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_complete_cli_persistence_exports_and_report(cli_setup: tuple[Path, Path, Path]) -> None:
    config, db, model = cli_setup
    run = cli(db, "run", str(config))
    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout)
    assert summary["status"] == "completed"
    experiment_id = summary["id"]
    result_dir = Path(summary["results"])
    experiment = json.loads((result_dir / "experiment.json").read_text())
    assert (
        experiment["metadata"]["model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    )
    assert experiment["config"]["mode"] == "synthetic_warm"
    assert "synthetic" in experiment["metadata"]["workload"]
    assert "ttft_ms" in experiment["metadata"]["unavailable_metrics"]
    assert len(experiment["cases"]) == 2
    assert all(len(case["samples"]) == 5 for case in experiment["cases"])
    assert all(case["process"]["upstream"]["TEST_ONLY"] for case in experiment["cases"])
    assert "no request latency or TTFT" in (result_dir / "report.md").read_text()
    assert "<svg" in (result_dir / "report.svg").read_text()
    with (result_dir / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert all(row["case_status"] == "completed" for row in rows)

    shown = cli(db, "show", experiment_id)
    assert shown.returncode == 0
    assert json.loads(shown.stdout) == experiment
    listed = cli(db, "list")
    assert json.loads(listed.stdout)[0]["id"] == experiment_id
    for format_name in ("csv", "json"):
        target = db.parent / f"manual-export.{format_name}"
        exported = cli(
            db, "export", experiment_id, "--format", format_name, "--output", str(target)
        )
        assert exported.returncode == 0, exported.stderr
        assert target.is_file()
    report_path = db.parent / "manual-report.md"
    reported = cli(db, "report", experiment_id, "--output", str(report_path))
    assert reported.returncode == 0, reported.stderr
    assert report_path.is_file()


@pytest.mark.parametrize("behavior", ["failure", "malformed"])
def test_failed_process_or_parser_is_persisted(
    cli_setup: tuple[Path, Path, Path], behavior: str
) -> None:
    config, db, _ = cli_setup
    result = cli(db, "run", str(config), behavior=behavior)
    assert result.returncode == 1, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "failed"
    experiment = Store(db).get_experiment(summary["id"])
    assert all(case["status"] == "failed" for case in experiment["cases"])
    assert all(case["samples"] == [] for case in experiment["cases"])
    assert all(case["error"] for case in experiment["cases"])
    if behavior == "failure":
        assert all(case["process"]["returncode"] == 7 for case in experiment["cases"])
        assert "deliberate engine failure" in experiment["cases"][0]["process"]["stderr"]
    else:
        assert "not valid JSON" in experiment["cases"][0]["process"]["stdout"]


def test_timeout_is_persisted_as_timeout(cli_setup: tuple[Path, Path, Path]) -> None:
    config, db, _ = cli_setup
    values = yaml.safe_load(config.read_text())
    values.update(timeout_s=0.15, generation_tokens=[])
    config.write_text(yaml.safe_dump(values))
    result = cli(db, "run", str(config), behavior="wait")
    assert result.returncode == 1, result.stderr
    experiment = Store(db).get_experiment(json.loads(result.stdout)["id"])
    assert experiment["status"] == "failed"
    assert experiment["cases"][0]["status"] == "timeout"
    assert experiment["cases"][0]["samples"] == []
    assert "process started" in experiment["cases"][0]["process"]["stdout"]


def test_sigint_cancels_cli_and_cleans_engine(cli_setup: tuple[Path, Path, Path]) -> None:
    config, db, _ = cli_setup
    marker = db.parent / "test-only-engine.pid"
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.cli", "--db", str(db), "run", str(config)],
        cwd=REPO,
        env=os.environ
        | {
            "LIOL_TEST_BEHAVIOR": "wait",
            "LIOL_TEST_READY_FILE": str(marker),
            "TMPDIR": str(db.parent),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"CLI exited before test engine started: {stdout}\n{stderr}")
            time.sleep(0.02)
        assert marker.exists(), "Test engine did not start"
        engine_pid = int(marker.read_text())
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stderr
        experiment = Store(db).get_experiment(json.loads(stdout)["id"])
        assert experiment["status"] == "canceled"
        assert all(case["status"] == "canceled" for case in experiment["cases"])
        assert not psutil.pid_exists(engine_pid)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)


def test_plan_does_not_execute_or_create_database(cli_setup: tuple[Path, Path, Path]) -> None:
    config, db, _ = cli_setup
    result = cli(db, "plan", str(config), behavior="failure")
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["cases"]) == 2
    assert not db.exists()


def test_cli_validation_and_unknown_ids(cli_setup: tuple[Path, Path, Path]) -> None:
    config, db, _ = cli_setup
    config.write_text("name: invalid\nmodel: nowhere\nbinary: nowhere\nunknown: true\n")
    invalid = cli(db, "run", str(config))
    assert invalid.returncode == 2
    assert "Unsupported configuration keys" in invalid.stderr
    assert not db.exists()
    missing = cli(db, "show", "nonexistent")
    assert missing.returncode == 2
    assert "Unknown experiment" in missing.stderr
    assert "Traceback" not in missing.stderr
