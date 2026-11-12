"""Warm-request CLI integration with a visibly isolated test HTTP server process."""

from __future__ import annotations

import csv
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
def request_setup(tmp_path: Path) -> tuple[Path, Path]:
    engine = tmp_path / "TEST_ONLY_llama-server"
    shutil.copyfile(REPO / "tests/fixtures/fake_llama_server.py", engine)
    engine.chmod(0o700)
    model = tmp_path / "TEST_ONLY_header.gguf"
    model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("TEST ONLY text fixture. " * 100)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "TEST ONLY HTTP request integration",
                "mode": "request_warm",
                "model": str(model),
                "binary": str(engine),
                "fixtures": [str(fixture)],
                "prompt_tokens": [8],
                "generation_tokens": [4],
                "threads": [1],
                "gpu_layers": [0],
                "batch_sizes": [32],
                "ubatch_sizes": [16],
                "context_size": 64,
                "repetitions": 5,
                "warmups": 1,
                "timeout_s": 2,
                "startup_timeout_s": 5,
            }
        )
    )
    return config, tmp_path / "results.sqlite3"


def run_cli(db: Path, *args: str, behavior: str = "success") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "backend.cli", "--db", str(db), *args],
        cwd=REPO,
        env=os.environ | {"TMPDIR": str(db.parent), "LIOL_SERVER_TEST_BEHAVIOR": behavior},
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )


def test_request_plan_and_complete_cli_exports(request_setup: tuple[Path, Path]) -> None:
    config, db = request_setup
    plan = run_cli(db, "plan", str(config))
    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan.stdout)["config"]["mode"] == "request_warm"
    assert not db.exists()
    completed = run_cli(db, "run", str(config))
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "completed"
    result_dir = Path(summary["results"])
    experiment = json.loads((result_dir / "experiment.json").read_text())
    assert experiment["metadata"]["mode"] == "request_warm"
    case = experiment["cases"][0]
    assert case["samples"] == []
    assert len(case["request_samples"]) == 5
    assert len(case["process"]["warmup_request_samples"]) == 1
    assert case["process"]["tokenizer"]["prompt_token_ids"] == list(range(8))
    assert all(sample["generated_tokens"] == 4 for sample in case["request_samples"])
    assert all(sample["ttft_ms"] >= 10 for sample in case["request_samples"])
    assert all(sample["cache_reused_tokens"] == 0 for sample in case["request_samples"])
    assert all(sample["request"]["cache_prompt"] is False for sample in case["request_samples"])
    assert all(sample["early_stop"] is False for sample in case["request_samples"])
    assert case["process"]["server"]["returncode"] is not None
    with (result_dir / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert all(row["request_latency_ms"] for row in rows)
    assert all(row["generated_tokens"] == "4" for row in rows)
    report = (result_dir / "report.md").read_text()
    assert "TTFT" in report
    assert "Synthetic prefill/decode microbenchmark" not in report
    assert (result_dir / "report.svg").exists()
    shown = run_cli(db, "show", summary["id"])
    assert shown.returncode == 0
    assert json.loads(shown.stdout) == experiment
    published = db.parent / "published"
    exported = subprocess.run(
        [
            sys.executable,
            "scripts/publish_results.py",
            summary["id"],
            str(published),
            "--db",
            str(db),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert exported.returncode == 0, exported.stderr
    public_text = published.with_suffix(".json").read_text()
    assert str(db.parent) not in public_text
    public_case = json.loads(public_text)["cases"][0]
    assert public_case["request_samples"][0]["request"]["prompt"] == list(range(8))
    assert len(public_case["request_samples"]) == 5


@pytest.mark.parametrize("behavior", ["failure", "malformed", "short_fixture", "cache_reuse"])
def test_request_failure_cases_have_no_accepted_samples(
    request_setup: tuple[Path, Path], behavior: str
) -> None:
    config, db = request_setup
    result = run_cli(db, "run", str(config), behavior=behavior)
    assert result.returncode == 1, result.stderr
    experiment = Store(db).get_experiment(json.loads(result.stdout)["id"])
    assert experiment["status"] == "failed"
    case = experiment["cases"][0]
    assert case["status"] == "failed"
    assert case["request_samples"] == []
    assert case["error"]
    assert case["process"]["server"]["returncode"] is not None


def test_request_timeout_is_saved_and_server_is_stopped(request_setup: tuple[Path, Path]) -> None:
    config, db = request_setup
    values = yaml.safe_load(config.read_text())
    values["timeout_s"] = 0.15
    config.write_text(yaml.safe_dump(values))
    result = run_cli(db, "run", str(config), behavior="wait")
    assert result.returncode == 1, result.stderr
    experiment = Store(db).get_experiment(json.loads(result.stdout)["id"])
    assert experiment["cases"][0]["status"] == "timeout"
    assert experiment["cases"][0]["request_samples"] == []
    assert experiment["cases"][0]["process"]["server"]["returncode"] is not None


def test_early_eos_counts_are_preserved_and_flagged(request_setup: tuple[Path, Path]) -> None:
    config, db = request_setup
    result = run_cli(db, "run", str(config), behavior="early_eos")
    assert result.returncode == 0, result.stderr
    experiment = Store(db).get_experiment(json.loads(result.stdout)["id"])
    samples = experiment["cases"][0]["request_samples"]
    assert len(samples) == 5
    assert all(sample["generated_tokens"] == 2 for sample in samples)
    assert all(sample["target_generated_tokens"] == 4 for sample in samples)
    assert all(sample["early_stop"] is True for sample in samples)
    assert all(sample["finish_reason"] == "eos" for sample in samples)


def test_sigint_cancels_requests_and_queued_cases_and_cleans_child(
    request_setup: tuple[Path, Path],
) -> None:
    config, db = request_setup
    values = yaml.safe_load(config.read_text())
    values["prompt_tokens"] = [8, 16]
    config.write_text(yaml.safe_dump(values))
    marker = db.parent / "TEST_ONLY_request_ready.pid"
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.cli", "--db", str(db), "run", str(config)],
        cwd=REPO,
        env=os.environ
        | {
            "TMPDIR": str(db.parent),
            "LIOL_SERVER_TEST_BEHAVIOR": "wait",
            "LIOL_SERVER_TEST_REQUEST_READY": str(marker),
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
                pytest.fail(f"CLI exited before request started: {stdout}\n{stderr}")
            time.sleep(0.02)
        assert marker.exists(), "Test request did not start"
        child_pid = int(marker.read_text())
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stderr
        experiment = Store(db).get_experiment(json.loads(stdout)["id"])
        assert experiment["status"] == "canceled"
        assert all(case["status"] == "canceled" for case in experiment["cases"])
        assert all(case["request_samples"] == [] for case in experiment["cases"])
        assert not psutil.pid_exists(child_pid)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
