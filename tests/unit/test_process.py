from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from backend.engines.process import ManagedProcess


def test_captures_both_streams_without_pipe_deadlock() -> None:
    result = ManagedProcess().run(
        [
            sys.executable,
            "-c",
            "import sys; print('x' * 200000); print('y' * 200000, file=sys.stderr)",
        ],
        timeout_s=5,
    )
    assert result.status == "completed"
    assert len(result.stdout.strip()) == 200000
    assert len(result.stderr.strip()) == 200000
    assert result.returncode == 0
    assert result.wall_ns > 0


def test_nonzero_exit_is_a_failure() -> None:
    result = ManagedProcess().run([sys.executable, "-c", "raise SystemExit(7)"], timeout_s=5)
    assert result.status == "failed"
    assert result.returncode == 7


def test_timeout_preserves_partial_output() -> None:
    result = ManagedProcess().run(
        [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"],
        timeout_s=0.2,
    )
    assert result.status == "timeout"
    assert "started" in result.stdout
    assert result.returncode != 0
    assert result.wall_ns < 5_000_000_000


def test_cancel_during_run() -> None:
    cancel = threading.Event()
    timer = threading.Timer(0.15, cancel.set)
    timer.start()
    try:
        result = ManagedProcess().run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_s=5,
            cancel=cancel,
        )
    finally:
        timer.cancel()
    assert result.status == "canceled"
    assert result.returncode != 0


def test_pre_canceled_run_never_starts(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    cancel = threading.Event()
    cancel.set()
    result = ManagedProcess().run(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        timeout_s=2,
        cancel=cancel,
    )
    assert result.status == "canceled"
    assert result.resource_samples == 0
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group behavior")
def test_timeout_kills_descendant_that_ignores_sigterm(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    child = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
    )
    result = ManagedProcess().run([sys.executable, "-c", parent], timeout_s=0.5)
    assert result.status == "timeout"
    assert marker.exists()
    child_pid = int(marker.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not psutil.pid_exists(child_pid):
            break
        if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.02)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("Descendant survived process-group shutdown")


def test_keyboard_interrupt_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    original_sleep = time.sleep
    interrupted = False

    def interrupt(_: float) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        original_sleep(_)

    monkeypatch.setattr("backend.engines.process.time.sleep", interrupt)
    result = ManagedProcess().run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_s=5,
    )
    assert result.status == "canceled"
    assert result.returncode != 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group behavior")
def test_successful_leader_does_not_leave_background_children(tmp_path: Path) -> None:
    marker = tmp_path / "background.pid"
    child = "import time; time.sleep(30)"
    parent = (
        "import subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"open({str(marker)!r}, 'w').write(str(child.pid)); time.sleep(0.1)"
    )
    result = ManagedProcess().run([sys.executable, "-c", parent], timeout_s=3)
    assert result.status == "completed"
    child_pid = int(marker.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not psutil.pid_exists(child_pid):
            return
        if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
            return
        time.sleep(0.02)
    os.kill(child_pid, signal.SIGKILL)
    pytest.fail("Background child survived successful leader exit")


def test_reports_sampled_process_memory() -> None:
    result = ManagedProcess().run(
        [sys.executable, "-c", "import time; allocation = bytearray(2000000); time.sleep(0.15)"],
        timeout_s=3,
    )
    assert result.status == "completed"
    assert result.resource_samples > 0
    assert result.sampled_peak_rss_bytes is not None
    assert result.sampled_peak_rss_bytes > 0
    assert result.sampled_cpu_seconds is not None
    assert result.sampled_cpu_seconds >= 0


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        ManagedProcess().run([sys.executable], timeout_s=timeout)


def test_missing_executable() -> None:
    with pytest.raises(FileNotFoundError):
        ManagedProcess().run(["/does/not/exist"], timeout_s=1)


def test_argument_values_do_not_invoke_a_shell() -> None:
    argument = "$(touch /tmp/liol-unwanted-shell-command); echo example"
    result = ManagedProcess().run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", argument],
        timeout_s=3,
    )
    assert result.stdout.strip() == argument
