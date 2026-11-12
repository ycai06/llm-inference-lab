"""Run an owned process without pipe deadlocks or orphaned child processes."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Literal

import psutil


@dataclass(frozen=True)
class ProcessResult:
    argv: list[str]
    stdout: str
    stderr: str
    returncode: int
    wall_ns: int
    status: Literal["completed", "failed", "timeout", "canceled"]
    sampled_peak_rss_bytes: int | None
    sampled_cpu_seconds: float | None
    resource_samples: int


class ManagedProcess:
    """Own one process group; resource samples refer only to its root process."""

    poll_interval_s = 0.02
    shutdown_grace_s = 0.5

    @staticmethod
    def _signal_group(process: subprocess.Popen[bytes], sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass

    def _stop(self, process: subprocess.Popen[bytes]) -> None:
        self._signal_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=self.shutdown_grace_s)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # The leader can exit while a descendant ignores SIGTERM.
            self._signal_group(process, signal.SIGKILL)
            process.wait()

    def run(
        self,
        argv: list[str],
        timeout_s: float,
        cancel: threading.Event | None = None,
    ) -> ProcessResult:
        if not argv or not all(isinstance(value, str) and value for value in argv):
            raise ValueError("argv must contain nonempty string arguments")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")

        started = time.perf_counter_ns()
        if cancel is not None and cancel.is_set():
            return ProcessResult(
                list(argv),
                "",
                "",
                -signal.SIGTERM,
                time.perf_counter_ns() - started,
                "canceled",
                None,
                None,
                0,
            )
        peak_rss: int | None = None
        cpu_seconds: float | None = None
        samples = 0
        status: Literal["completed", "failed", "timeout", "canceled"] = "completed"
        # Regular files cannot fill a pipe buffer while the other stream blocks.
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=os.name == "posix",
                shell=False,
            )
            try:
                try:
                    monitored = psutil.Process(process.pid)
                except psutil.NoSuchProcess:
                    monitored = None
                while process.poll() is None:
                    if monitored is not None:
                        try:
                            rss = monitored.memory_info().rss
                            cpu = monitored.cpu_times()
                            peak_rss = max(peak_rss or 0, rss)
                            cpu_seconds = max(cpu_seconds or 0.0, cpu.user + cpu.system)
                            samples += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    if cancel is not None and cancel.is_set():
                        status = "canceled"
                        self._stop(process)
                        break
                    if (time.perf_counter_ns() - started) / 1e9 >= timeout_s:
                        status = "timeout"
                        self._stop(process)
                        break
                    if cancel is not None:
                        cancel.wait(self.poll_interval_s)
                    else:
                        time.sleep(self.poll_interval_s)
            except KeyboardInterrupt:
                status = "canceled"
                self._stop(process)
            except BaseException:
                self._stop(process)
                raise
            finally:
                if process.poll() is None:
                    self._stop(process)
                elif os.name == "posix":
                    # An exited group leader may still have background children.
                    self._signal_group(process, signal.SIGKILL)
            returncode = process.wait()
            wall_ns = time.perf_counter_ns() - started
            if status == "completed" and returncode != 0:
                status = "failed"
            stdout_file.seek(0)
            stderr_file.seek(0)
            return ProcessResult(
                list(argv),
                stdout_file.read().decode("utf-8", errors="replace"),
                stderr_file.read().decode("utf-8", errors="replace"),
                returncode,
                wall_ns,
                status,
                peak_rss,
                cpu_seconds,
                samples,
            )
