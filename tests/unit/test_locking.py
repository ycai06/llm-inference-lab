from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from backend.benchmarks.locking import benchmark_lock


def test_second_lock_is_rejected_and_released_after_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.benchmarks.locking.tempfile.gettempdir", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="intentional test exception"):
        with benchmark_lock():
            with pytest.raises(RuntimeError, match="Another LIOL benchmark"):
                with benchmark_lock():
                    pytest.fail("Duplicate benchmark acquired the lock")
            raise ValueError("intentional test exception")
    with benchmark_lock():
        pass


def test_lock_coordinates_separate_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.benchmarks.locking.tempfile.gettempdir", lambda: str(tmp_path))
    child = (
        "from backend.benchmarks import locking; "
        f"locking.tempfile.gettempdir = lambda: {str(tmp_path)!r}; "
        "\nwith locking.benchmark_lock():\n    print('acquired')\n"
    )
    with benchmark_lock():
        rejected = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, timeout=5, check=False
        )
    assert rejected.returncode != 0
    assert "Another LIOL benchmark" in rejected.stderr
    accepted = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=5, check=False
    )
    assert accepted.returncode == 0
    assert accepted.stdout.strip() == "acquired"
