"""Best-effort hardware reads; unavailable counters remain explicit."""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil


def command_output(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment() -> dict[str, Any]:
    disk = shutil.disk_usage(Path.cwd())
    return {
        "os": platform.system(),
        "os_version": command_output(["sw_vers", "-productVersion"])
        if platform.system() == "Darwin"
        else platform.release(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "soc": command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if platform.system() == "Darwin"
        else platform.processor(),
        "physical_memory_bytes": psutil.virtual_memory().total,
        "logical_cpu_count": psutil.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "disk_available_bytes": disk.free,
        "python": platform.python_version(),
        "power_source_observation": command_output(["pmset", "-g", "batt"])
        if platform.system() == "Darwin"
        else None,
        "thermal_pressure": None,
        "gpu_utilization": None,
        "gpu_memory_bytes": None,
        "power_watts": None,
    }
