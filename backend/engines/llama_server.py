"""Own a single authenticated localhost llama-server and measure native requests."""

from __future__ import annotations

import hashlib
import math
import os
import re
import secrets
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

import psutil

from backend.engines.http_stream import HTTPStatusError, LocalHTTPClient, SSEEvent, StreamResponse
from backend.engines.process import ManagedProcess


@dataclass(frozen=True)
class ServerSettings:
    model: Path
    threads: int
    gpu_layers: int
    batch_size: int
    ubatch_size: int
    context_size: int
    flash_attention: str = "auto"
    startup_timeout_s: float = 120

    def validate(self) -> None:
        for name in ("threads", "batch_size", "ubatch_size", "context_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.gpu_layers) is not int or self.gpu_layers < -1:
            raise ValueError("gpu_layers must be -1 or nonnegative")
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        if self.flash_attention not in ("on", "off", "auto"):
            raise ValueError("flash_attention must be on, off, or auto")
        if not math.isfinite(self.startup_timeout_s) or self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive and finite")
        if not self.model.is_file():
            raise ValueError(f"Model file does not exist: {self.model}")
        with self.model.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError(f"Model is not a GGUF file: {self.model}")


def _supports(help_text: str, flag: str) -> bool:
    return re.search(r"(?<!\S)" + re.escape(flag) + r"(?=[\s,]|$)", help_text) is not None


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _token_ids(value: Any, name: str, empty: bool = True) -> list[int]:
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{name} must be a {'nonempty ' if not empty else ''}list of token IDs")
    for token in value:
        _integer(token, name)
    return list(value)


def _event_record(event: SSEEvent, started_ns: int) -> dict[str, Any]:
    data = event.data if isinstance(event.data, dict) else {}
    return {
        "arrival_ns": event.arrival_ns - started_ns,
        "tokens": data.get("tokens", []),
        "content": data.get("content", ""),
        "data": event.data,
        "raw_data": event.raw_data,
    }


def analyze_stream(
    stream: StreamResponse, request: dict[str, Any], trial_index: int = 0
) -> dict[str, Any]:
    """Validate the native protocol; preserve engine and transport timing separately."""
    if not stream.events or not isinstance(stream.events[-1].data, dict):
        raise ValueError("No final native completion response")
    final = stream.events[-1].data
    if final.get("stop") is not True:
        raise ValueError("Missing terminal completion event")
    tokens = _token_ids(request["prompt"], "request prompt", empty=False)
    target = _integer(request["n_predict"], "n_predict", 1)
    actual = _integer(final.get("tokens_predicted"), "tokens_predicted")
    if actual > target:
        raise ValueError("Server generated more tokens than requested")
    if _integer(final.get("tokens_evaluated"), "tokens_evaluated") != len(tokens):
        raise ValueError("Observed prompt count differs from supplied token IDs")
    if final.get("truncated") is not False:
        raise ValueError("Server truncated the context or omitted the truncation flag")
    if _token_ids(final.get("tokens"), "terminal tokens") or final.get("content") != "":
        raise ValueError(
            "Unsupported terminal content: native stream must carry output in partials"
        )
    finish = final.get("stop_type")
    if finish not in ("eos", "limit", "word"):
        raise ValueError("Unsupported or missing stop_type")
    settings = final.get("generation_settings")
    if not isinstance(settings, dict):
        raise ValueError("Missing generation_settings")
    for field in ("n_predict", "seed", "temperature", "ignore_eos", "samplers"):
        value = settings.get(field)
        if field in ("n_predict", "seed") and type(value) is not int:
            raise ValueError(f"Server returned invalid {field}")
        if field == "ignore_eos" and type(value) is not bool:
            raise ValueError("Server returned invalid ignore_eos")
        if field == "temperature":
            _number(value, "generation_settings.temperature")
        if value != request[field]:
            raise ValueError(f"Server changed requested {field}")
    timings = final.get("timings")
    if not isinstance(timings, dict):
        raise ValueError("Missing native timing fields")
    cached = _integer(timings.get("cache_n"), "timings.cache_n")
    processed = _integer(timings.get("prompt_n"), "timings.prompt_n")
    predicted = _integer(timings.get("predicted_n"), "timings.predicted_n")
    if processed + cached != len(tokens) or predicted != actual:
        raise ValueError("Native timing token counts do not match observed request counts")
    if not request["cache_prompt"] and cached:
        raise ValueError("Unexpected prefix-cache reuse while cache_prompt is disabled")
    prompt_ms = _number(timings.get("prompt_ms"), "timings.prompt_ms")
    decode_ms = _number(timings.get("predicted_ms"), "timings.predicted_ms")
    prefill = _number(timings.get("prompt_per_second"), "timings.prompt_per_second")
    decode = _number(timings.get("predicted_per_second"), "timings.predicted_per_second")
    if processed:
        if not prompt_ms or not math.isclose(
            prefill, processed * 1000 / prompt_ms, rel_tol=0.005, abs_tol=0.01
        ):
            raise ValueError("Inconsistent native prefill timing")
    if actual >= 2:
        if not decode_ms or not math.isclose(
            decode, (actual - 1) * 1000 / decode_ms, rel_tol=0.005, abs_tol=0.01
        ):
            raise ValueError("Inconsistent native decode timing")
    warnings: list[str] = []
    arrivals, content, observed_ids = [], [], []
    last_decoded = 0
    complete_coverage = True
    for event in stream.events[:-1]:
        data = event.data
        if data is None or data == "[DONE]":
            continue
        if not isinstance(data, dict):
            raise ValueError("SSE data must be a native response object")
        if "prompt_progress" in data:
            continue
        ids = _token_ids(data.get("tokens", []), "stream tokens")
        text = data.get("content", "")
        if not isinstance(text, str):
            raise ValueError("Stream content must be a string")
        if not ids:
            if text:
                raise ValueError("Generated content arrived without requested token IDs")
            continue
        if data.get("stop") is not False:
            raise ValueError("Invalid partial completion stop flag")
        count = _integer(data.get("tokens_predicted"), "partial tokens_predicted", 1)
        if count <= last_decoded or count > actual:
            raise ValueError("Nonmonotonic or inconsistent streamed token count")
        complete_coverage &= count - last_decoded == len(ids) == 1
        arrivals.append(event.arrival_ns)
        observed_ids.extend(ids)
        content.append(text)
        last_decoded = count
    complete_coverage &= last_decoded == actual and len(observed_ids) == actual
    if not complete_coverage:
        warnings.append(
            "Stream events do not cover every generated token; inter-token latency unavailable"
        )
    unique_arrivals = len(arrivals) == len(set(arrivals))
    if not unique_arrivals:
        warnings.append(
            "Multiple token events shared a receive timestamp; inter-token latency unavailable"
        )
    intervals = [
        (later - earlier) / 1e6 for earlier, later in zip(arrivals, arrivals[1:], strict=False)
    ]
    if intervals and any(value <= 0 for value in intervals):
        if unique_arrivals:
            raise ValueError("Nonmonotonic receive timestamps")
    valid_intervals = intervals if complete_coverage and unique_arrivals else []
    if actual < target:
        warnings.append(f"Generation ended early: {actual} of {target} requested tokens")
    if actual and not arrivals:
        warnings.append("No generated token event observed; TTFT unavailable")
    if stream.completed_ns <= stream.started_ns or any(
        not stream.started_ns <= arrival <= stream.completed_ns for arrival in arrivals
    ):
        raise ValueError("Invalid request timing boundaries")
    return {
        "trial_index": _integer(trial_index, "trial_index"),
        "request_latency_ms": (stream.completed_ns - stream.started_ns) / 1e6,
        "ttft_ms": (arrivals[0] - stream.started_ns) / 1e6 if arrivals else None,
        "prompt_tokens": len(tokens),
        "prompt_processed_tokens": processed,
        "generated_tokens": actual,
        "target_generated_tokens": target,
        "prefill_tokens_per_second": prefill if processed and prefill > 0 else None,
        "decode_tokens_per_second": decode if actual >= 2 and decode > 0 else None,
        "mean_inter_token_latency_ms": statistics.mean(valid_intervals)
        if valid_intervals
        else None,
        "median_inter_token_latency_ms": statistics.median(valid_intervals)
        if valid_intervals
        else None,
        "cache_reused_tokens": cached,
        "early_stop": actual < target,
        "finish_reason": finish,
        "output": "".join(content),
        "events": [_event_record(event, stream.started_ns) for event in stream.events],
        "request": request,
        "response": final,
        "warnings": warnings,
    }


class LlamaServer:
    def __init__(self, binary: Path) -> None:
        self.binary = binary.expanduser().resolve()
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise ValueError(f"Engine binary is missing or not executable: {self.binary}")
        self._probe: dict[str, Any] | None = None

    def probe(self) -> dict[str, Any]:
        if self._probe is not None:
            return self._probe
        process = ManagedProcess()
        outputs = {}
        for key, flag in (
            ("help", "--help"),
            ("version", "--version"),
            ("devices", "--list-devices"),
        ):
            result = process.run([str(self.binary), flag], timeout_s=15)
            if result.status != "completed":
                raise ValueError(f"Cannot inspect llama-server {flag}: {result.stderr.strip()}")
            outputs[key] = result.stdout + result.stderr
        required = (
            "--model",
            "--host",
            "--port",
            "--threads",
            "--threads-batch",
            "--gpu-layers",
            "--device",
            "--ctx-size",
            "--batch-size",
            "--ubatch-size",
            "--flash-attn",
            "--fit",
            "--parallel",
            "--no-cache-prompt",
            "--cache-reuse",
            "--cache-ram",
            "--no-cache-idle-slots",
            "--no-context-shift",
            "--no-ui",
            "--offline",
            "--perf",
            "--api-key-file",
            "--no-kv-offload",
            "--no-op-offload",
        )
        missing = [flag for flag in required if not _supports(outputs["help"], flag)]
        if missing:
            raise ValueError("llama-server lacks required switches: " + ", ".join(missing))
        digest = hashlib.sha256()
        with self.binary.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self._probe = dict(outputs, binary_sha256=digest.hexdigest())
        return self._probe

    def session(self, settings: ServerSettings, cancel: threading.Event) -> ManagedServer:
        return ManagedServer(self, settings, cancel)


class ManagedServer:
    def __init__(
        self, engine: LlamaServer, settings: ServerSettings, cancel: threading.Event
    ) -> None:
        self.engine, self.settings, self.cancel = engine, settings, cancel
        self.metadata: dict[str, Any] = {"status": "not_started"}
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout: BinaryIO | None = None
        self._stderr: BinaryIO | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._api_key = ""
        self._client: LocalHTTPClient | None = None
        self._started_ns = 0
        self._monitor_stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._peak_rss: int | None = None
        self._cpu_seconds: float | None = None
        self._samples = 0
        self._entered = False
        self._closed = False

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _read_log(self, stream: BinaryIO | None) -> str:
        if stream is None:
            return ""
        size = os.fstat(stream.fileno()).st_size
        limit = 16 * 1024 * 1024
        if size > limit:
            self.metadata["logs_truncated"] = True
        # pread never changes the file offset shared with the child's output descriptor.
        text = os.pread(stream.fileno(), limit, 0).decode("utf-8", errors="replace")
        return text.replace(self._api_key, "<redacted>") if self._api_key else text

    def _refresh(self) -> None:
        self.metadata.update(
            stdout=self._read_log(self._stdout),
            stderr=self._read_log(self._stderr),
            sampled_peak_rss_bytes=self._peak_rss,
            sampled_cpu_seconds=self._cpu_seconds,
            resource_samples=self._samples,
        )
        if self._process is not None:
            self.metadata["returncode"] = self._process.poll()
            self.metadata["wall_ns"] = time.perf_counter_ns() - self._started_ns

    def _sample(self) -> None:
        assert self._process is not None
        try:
            process = psutil.Process(self._process.pid)
        except psutil.NoSuchProcess:
            return
        while not self._monitor_stop.is_set():
            try:
                rss, cpu = process.memory_info().rss, process.cpu_times()
                self._peak_rss = max(self._peak_rss or 0, rss)
                self._cpu_seconds = max(self._cpu_seconds or 0, cpu.user + cpu.system)
                self._samples += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                if not self._alive():
                    return
            self._monitor_stop.wait(0.02)

    def __enter__(self) -> ManagedServer:
        if self._entered:
            raise RuntimeError("A managed server session cannot be entered twice")
        self._entered = True
        self.settings.validate()
        if self.cancel.is_set():
            raise InterruptedError("Canceled before server startup")
        probe = self.engine.probe()
        cpu = self.settings.gpu_layers == 0
        device = "none"
        if not cpu:
            match = re.search(r"^\s*(MTL\d+):", probe["devices"], re.MULTILINE)
            if match is None:
                raise ValueError("Metal requested but no MTL device was reported by the engine")
            device = match.group(1)
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="liol-server-")
            key_path = Path(self._temporary.name) / "api-key"
            self._api_key = secrets.token_urlsafe(32)
            descriptor = os.open(key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w") as key_file:
                key_file.write(self._api_key + "\n")
            s = self.settings
            argv = [
                str(self.engine.binary),
                "--model",
                str(s.model.resolve()),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--threads",
                str(s.threads),
                "--threads-batch",
                str(s.threads),
                "--gpu-layers",
                "all" if s.gpu_layers == -1 else str(s.gpu_layers),
                "--device",
                device,
                "--ctx-size",
                str(s.context_size),
                "--batch-size",
                str(s.batch_size),
                "--ubatch-size",
                str(s.ubatch_size),
                "--flash-attn",
                s.flash_attention,
                "--fit",
                "off",
                "--parallel",
                "1",
                "--no-cache-prompt",
                "--cache-reuse",
                "0",
                "--cache-ram",
                "0",
                "--no-cache-idle-slots",
                "--no-context-shift",
                "--no-ui",
                "--offline",
                "--perf",
                "--api-key-file",
                str(key_path),
            ]
            if cpu:
                argv += ["--no-kv-offload", "--no-op-offload"]
            for optional in ("--no-slots", "--no-cont-batching"):
                if _supports(probe["help"], optional):
                    argv.append(optional)
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("LLAMA_ARG_", "AIP_")) and k != "LLAMA_API_KEY"
            }
            redacted = ["<private-api-key-file>" if arg == str(key_path) else arg for arg in argv]
            settings = asdict(s)
            settings["model"] = str(s.model.resolve())
            self.metadata.update(
                engine=probe,
                argv=redacted,
                settings=settings,
                env_policy="LLAMA_ARG_*, LLAMA_API_KEY and AIP_* removed",
                status="starting",
                pid=None,
                logs_truncated=False,
            )
            self._stdout, self._stderr = tempfile.TemporaryFile(), tempfile.TemporaryFile()
            self._started_ns = time.perf_counter_ns()
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=self._stdout,
                stderr=self._stderr,
                env=env,
                start_new_session=True,
            )
            self.metadata["pid"] = self._process.pid
            self._monitor = threading.Thread(
                target=self._sample, name="liol-server-resources", daemon=True
            )
            self._monitor.start()
            deadline = time.monotonic() + s.startup_timeout_s
            while True:
                if self.cancel.is_set():
                    raise InterruptedError("Canceled during server startup")
                if not self._alive():
                    raise RuntimeError("llama-server exited before readiness")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("llama-server did not become ready before startup deadline")
                logs = self._read_log(self._stdout) + self._read_log(self._stderr)
                match = re.search(r"listening on http://127\.0\.0\.1:(\d+)\b", logs)
                if match:
                    port = int(match.group(1))
                    self._client = LocalHTTPClient(port, self._api_key, self._alive)
                    self.metadata["address"] = f"http://127.0.0.1:{port}"
                    try:
                        health = self._client.json(
                            "GET", "/health", None, min(0.5, remaining), self.cancel
                        )
                        if health.get("status") == "ok":
                            props = self._client.json(
                                "GET",
                                "/props",
                                None,
                                min(2, max(0.01, deadline - time.monotonic())),
                                self.cancel,
                            )
                            self._validate_props(props)
                            self.metadata.update(
                                props=props,
                                status="ready",
                                startup_duration_ms=(time.perf_counter_ns() - self._started_ns)
                                / 1e6,
                            )
                            self._refresh()
                            return self
                    except HTTPStatusError as exc:
                        if exc.status != 503:
                            raise
                    except (ConnectionError, TimeoutError):
                        pass
                self.cancel.wait(min(0.05, max(0, deadline - time.monotonic())))
        except BaseException:
            self.metadata["status"] = "startup_failed"
            self.close()
            raise

    def _validate_props(self, props: dict[str, Any]) -> None:
        model = props.get("model_path")
        defaults = props.get("default_generation_settings")
        if not isinstance(model, str) or Path(model).resolve() != self.settings.model.resolve():
            raise ValueError("Server reports a different model path")
        if type(props.get("total_slots")) is not int or props["total_slots"] != 1:
            raise ValueError("Server did not retain the requested single slot")
        if not isinstance(defaults, dict) or defaults.get("n_ctx") != self.settings.context_size:
            raise ValueError("Server changed requested context size")
        if not isinstance(props.get("build_info"), str):
            raise ValueError("Server does not expose required build provenance")

    def tokenize(
        self,
        text: str,
        add_special: bool = True,
        parse_special: bool = False,
        timeout_s: float = 30,
    ) -> list[int]:
        if (
            not isinstance(text, str)
            or type(add_special) is not bool
            or type(parse_special) is not bool
        ):
            raise ValueError(
                "Tokenization requires text and explicit boolean special-token policies"
            )
        if self._client is None or not self._alive():
            raise RuntimeError("Managed server is not running")
        result = self._client.json(
            "POST",
            "/tokenize",
            {
                "content": text,
                "add_special": add_special,
                "parse_special": parse_special,
                "with_pieces": False,
            },
            timeout_s,
            self.cancel,
        )
        self._refresh()
        return _token_ids(result.get("tokens"), "tokenizer tokens")

    def complete(
        self,
        prompt_tokens: list[int],
        generation_tokens: int,
        seed: int,
        ignore_eos: bool,
        cache_prompt: bool,
        timeout_s: float,
        cancel: threading.Event,
        trial_index: int = 0,
    ) -> dict[str, Any]:
        prompt_tokens = _token_ids(prompt_tokens, "prompt_tokens", empty=False)
        _integer(generation_tokens, "generation_tokens", 1)
        _integer(seed, "seed")
        if type(ignore_eos) is not bool or type(cache_prompt) is not bool:
            raise ValueError("ignore_eos and cache_prompt must be booleans")
        if len(prompt_tokens) + generation_tokens >= self.settings.context_size:
            raise ValueError("Prompt and requested generation must fit without context shifting")
        if self._client is None or not self._alive():
            raise RuntimeError("Managed server is not running")
        request = {
            "prompt": prompt_tokens,
            "n_predict": generation_tokens,
            "seed": seed,
            "temperature": 0.0,
            "samplers": ["temperature"],
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": [],
            "ignore_eos": ignore_eos,
            "cache_prompt": cache_prompt,
            "id_slot": 0,
            "stream": True,
            "return_tokens": True,
            "return_progress": False,
            "timings_per_token": False,
        }
        started = time.perf_counter_ns()
        try:
            stream = self._client.stream("/completion", request, timeout_s, cancel)
            return analyze_stream(stream, request, trial_index)
        except BaseException as exc:
            self.metadata["last_request_failure"] = {
                "error": str(exc),
                "request": request,
                "events": [_event_record(e, started) for e in self._client.last_events],
            }
            raise
        finally:
            self._refresh()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        self._monitor_stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=1)
        self._refresh()
        if self.metadata.get("status") != "startup_failed":
            self.metadata["status"] = "stopped"
        for stream in (self._stdout, self._stderr):
            if stream is not None:
                stream.close()
        self._stdout = self._stderr = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._client = None
        self._api_key = ""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
