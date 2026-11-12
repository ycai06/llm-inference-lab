"""Version-probed adapter for upstream synthetic-token llama-bench workloads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from backend.engines.process import ManagedProcess, ProcessResult


@dataclass(frozen=True)
class BenchCase:
    kind: Literal["prefill", "decode"]
    tokens: int
    threads: int
    gpu_layers: int
    batch_size: int
    ubatch_size: int
    flash_attention: str = "auto"

    def validate(self) -> None:
        if self.kind not in ("prefill", "decode"):
            raise ValueError("kind must be prefill or decode")
        for name in ("tokens", "threads", "batch_size", "ubatch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.gpu_layers) is not int or self.gpu_layers < -1:
            raise ValueError("gpu_layers must be -1 or a nonnegative integer")
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        if self.flash_attention not in ("auto", "on", "off"):
            raise ValueError("flash_attention must be auto, on, or off")


def _executable(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"Engine binary is missing or not executable: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _supports(help_text: str, flag: str) -> bool:
    return re.search(r"(?<!\S)" + re.escape(flag) + r"(?=[\s,]|$)", help_text) is not None


class LlamaBench:
    def __init__(self, binary: Path) -> None:
        self.binary = _executable(binary)
        self.process = ManagedProcess()
        self._probe: dict[str, Any] | None = None

    def probe(self) -> dict[str, Any]:
        if self._probe is not None:
            return self._probe
        help_result = self.process.run([str(self.binary), "--help"], timeout_s=15)
        if help_result.status != "completed":
            raise ValueError(f"Unable to inspect engine help: {help_result.stderr.strip()}")
        help_text = help_result.stdout + help_result.stderr
        if not _supports(help_text, "-m") or not _supports(help_text, "-r"):
            raise ValueError("Binary does not expose the required llama-bench interface")
        devices: str | None = None
        if _supports(help_text, "--list-devices"):
            device_result = self.process.run([str(self.binary), "--list-devices"], timeout_s=15)
            if device_result.status == "completed":
                devices = device_result.stdout + device_result.stderr
        version: str | None = None
        # Supported by the inspected local build, even when absent from its help.
        version_result = self.process.run([str(self.binary), "--version"], timeout_s=15)
        if version_result.status == "completed":
            version = (version_result.stdout + version_result.stderr).strip()
        self._probe = {
            "version": version,
            "help": help_text,
            "devices": devices,
            "binary_sha256": _sha256(self.binary),
        }
        return self._probe

    @staticmethod
    def parse(
        stdout: str,
        case: BenchCase,
        repetitions: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return parse_samples(stdout, case, repetitions)

    def run(
        self,
        model: Path,
        case: BenchCase,
        repetitions: int,
        timeout_s: float,
        cancel: threading.Event | None = None,
    ) -> ProcessResult:
        case.validate()
        if type(repetitions) is not int or repetitions <= 0:
            raise ValueError("repetitions must be a positive integer")
        model = model.expanduser().resolve()
        if not model.is_file():
            raise ValueError(f"Model file does not exist: {model}")
        with model.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ValueError(f"Model is not a GGUF file: {model}")
        probe = self.probe()
        required = (
            "-m",
            "-p",
            "-n",
            "-t",
            "-ngl",
            "-b",
            "-ub",
            "-fa",
            "-r",
            "-o",
            "-dev",
            "-nkvo",
            "-nopo",
        )
        missing = [flag for flag in required if not _supports(probe["help"], flag)]
        if missing:
            raise ValueError("Engine lacks required switches: " + ", ".join(missing))
        flash_line = next(
            (line for line in probe["help"].splitlines() if "--flash-attn" in line), ""
        )
        if not all(value in flash_line for value in ("on", "off", "auto")):
            raise ValueError("Engine flash-attention syntax is unsupported; require on|off|auto")
        device = "none"
        if case.gpu_layers != 0:
            match = re.search(r"\bMTL\d+\b", probe["devices"] or "")
            if match is None:
                raise ValueError("Metal requested but no MTL device was reported by the engine")
            device = match.group(0)
        cpu = case.gpu_layers == 0
        argv = [
            str(self.binary),
            "-m",
            str(model),
            "-o",
            "json",
            "-r",
            str(repetitions),
            "-p",
            str(case.tokens if case.kind == "prefill" else 0),
            "-n",
            str(case.tokens if case.kind == "decode" else 0),
            "-t",
            str(case.threads),
            "-ngl",
            str(case.gpu_layers),
            "-b",
            str(case.batch_size),
            "-ub",
            str(case.ubatch_size),
            "-fa",
            case.flash_attention,
            "-dev",
            device,
            "-nkvo",
            "1" if cpu else "0",
            "-nopo",
            "1" if cpu else "0",
        ]
        if _supports(probe["help"], "--offline"):
            argv.append("--offline")
        if _supports(probe["help"], "-v"):
            argv.append("-v")
        return self.process.run(argv, timeout_s=timeout_s, cancel=cancel)


def parse_samples(
    stdout: str,
    case: BenchCase,
    repetitions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Require one matching workload and authoritative per-repetition nanoseconds."""
    case.validate()
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    try:
        decoded = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Engine did not return valid JSON: {exc}") from exc
    if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], dict):
        raise ValueError("Expected exactly one llama-bench JSON result row")
    row = decoded[0]
    expected = {
        "n_prompt": case.tokens if case.kind == "prefill" else 0,
        "n_gen": case.tokens if case.kind == "decode" else 0,
        "n_threads": case.threads,
        "n_gpu_layers": case.gpu_layers,
        "n_batch": case.batch_size,
        "n_ubatch": case.ubatch_size,
        "n_depth": 0,
        # Verified against b12182 include/llama.h enum llama_flash_attn_type.
        "flash_attn": {"auto": -1, "off": 0, "on": 1}[case.flash_attention],
    }
    for field, value in expected.items():
        if type(row.get(field)) is not int or row[field] != value:
            raise ValueError(f"Engine result {field} does not match requested value {value}")
    cpu = case.gpu_layers == 0
    if type(row.get("no_kv_offload")) is not bool or row["no_kv_offload"] != cpu:
        raise ValueError("Engine no_kv_offload does not match requested configuration")
    if type(row.get("no_op_offload")) is not int or row["no_op_offload"] != int(cpu):
        raise ValueError("Engine no_op_offload does not match requested configuration")
    devices = row.get("devices")
    if cpu and devices != "none":
        raise ValueError("CPU workload did not report devices=none")
    if not cpu and (not isinstance(devices, str) or re.fullmatch(r"MTL\d+", devices) is None):
        raise ValueError("Metal workload did not report an explicit MTL device")
    durations = row.get("samples_ns")
    if not isinstance(durations, list) or len(durations) != repetitions:
        raise ValueError("Engine sample count does not match requested repetitions")
    reported_rates = row.get("samples_ts")
    if not isinstance(reported_rates, list) or len(reported_rates) != repetitions:
        raise ValueError("Engine samples_ts count does not match requested repetitions")
    samples: list[dict[str, Any]] = []
    for index, (duration, rate) in enumerate(zip(durations, reported_rates, strict=True)):
        if type(duration) is not int or duration <= 0:
            raise ValueError("Engine samples_ns must contain positive integer nanoseconds")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Engine samples_ts must contain finite positive rates")
        calculated = case.tokens * 1e9 / duration
        # Upstream rounds rates to six significant digits; integer ns is authoritative.
        if not math.isclose(float(rate), calculated, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Engine token rate is inconsistent with token count and duration")
        samples.append(
            {
                "trial_index": index,
                "duration_ns": duration,
                "tokens_per_second": calculated,
                "tokens": case.tokens,
            }
        )
    return samples, dict(row)
