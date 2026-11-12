"""Strict configuration for warm, tokenizer-validated text request experiments."""

from __future__ import annotations

import itertools
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.schemas.config import ConfigurationError

MAX_FIXTURE_BYTES = 256 * 1024
MAX_REQUEST_CASES = 128


@dataclass(frozen=True)
class RequestConfig:
    name: str
    model: Path
    binary: Path
    fixtures: tuple[Path, ...]
    prompt_tokens: tuple[int, ...] = (128, 512, 1024)
    generation_tokens: tuple[int, ...] = (64, 128, 256)
    threads: tuple[int, ...] = (6,)
    gpu_layers: tuple[int, ...] = (-1,)
    batch_sizes: tuple[int, ...] = (512,)
    ubatch_sizes: tuple[int, ...] = (128,)
    context_size: int = 2048
    repetitions: int = 5
    warmups: int = 1
    seed: int = 42
    ignore_eos: bool = False
    cache_prompt: bool = False
    timeout_s: float = 180
    startup_timeout_s: float = 120
    shuffle_seed: int = 42
    flash_attention: str = "auto"
    mode: str = "request_warm"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            model=str(self.model),
            binary=str(self.binary),
            fixtures=[str(path) for path in self.fixtures],
        )
        return result

    def cases(self) -> list[dict[str, Any]]:
        if _matrix_size(self) > MAX_REQUEST_CASES:
            raise ConfigurationError("Experiment matrix exceeds 128 request cases; split the suite")
        cases = [
            dict(
                kind="request",
                fixture=str(fixture),
                prompt_tokens=prompt,
                generation_tokens=generation,
                threads=threads,
                gpu_layers=gpu_layers,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                context_size=self.context_size,
                flash_attention=self.flash_attention,
            )
            for (
                fixture,
                prompt,
                generation,
                threads,
                gpu_layers,
                batch_size,
                ubatch_size,
            ) in itertools.product(
                self.fixtures,
                self.prompt_tokens,
                self.generation_tokens,
                self.threads,
                self.gpu_layers,
                self.batch_sizes,
                self.ubatch_sizes,
            )
        ]
        random.Random(self.shuffle_seed).shuffle(cases)
        return cases


def _matrix_size(config: RequestConfig) -> int:
    return math.prod(
        len(values)
        for values in (
            config.fixtures,
            config.prompt_tokens,
            config.generation_tokens,
            config.threads,
            config.gpu_layers,
            config.batch_sizes,
            config.ubatch_sizes,
        )
    )


def _path(value: Any, field: str, config_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a nonempty path")
    expanded = os.path.expandvars(value)
    if re.search(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
        raise ConfigurationError(f"Set the environment variable in {field}: {value}")
    if not expanded.strip():
        raise ConfigurationError(f"{field} must expand to a nonempty path")
    try:
        candidate = Path(expanded).expanduser()
        return (candidate if candidate.is_absolute() else config_path.parent / candidate).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid {field} path: {exc}") from exc


def _fixture(path: Path) -> None:
    try:
        if not path.is_file():
            raise ConfigurationError(f"Fixture is missing or not a regular file: {path}")
        # Bound the actual read as well as stat: another process can change the file.
        if path.stat().st_size > MAX_FIXTURE_BYTES:
            raise ConfigurationError(f"Fixture exceeds {MAX_FIXTURE_BYTES} bytes: {path}")
        with path.open("rb") as stream:
            data = stream.read(MAX_FIXTURE_BYTES + 1)
        if len(data) > MAX_FIXTURE_BYTES:
            raise ConfigurationError(f"Fixture exceeds {MAX_FIXTURE_BYTES} bytes: {path}")
        text = data.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"Cannot read UTF-8 fixture {path}: {exc}") from exc
    if not text.strip():
        raise ConfigurationError(f"Fixture must contain nonempty text: {path}")


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConfigurationError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def parse_request_config(path: Path, raw: dict[str, Any]) -> RequestConfig:
    """Validate parsed YAML without starting an engine or accessing model weights."""
    unknown = set(raw) - set(RequestConfig.__dataclass_fields__)
    if unknown:
        raise ConfigurationError(f"Unsupported configuration keys: {sorted(unknown, key=str)}")
    values = raw.copy()
    if not isinstance(values.get("name"), str) or not values["name"].strip():
        raise ConfigurationError("name must be a nonempty string")
    for field in ("model", "binary"):
        values[field] = _path(values.get(field), field, path)
    fixtures = values.get("fixtures")
    if not isinstance(fixtures, (list, tuple)) or not fixtures:
        raise ConfigurationError("fixtures must be a nonempty list of text fixture paths")
    resolved = tuple(_path(value, "fixture", path) for value in fixtures)
    if len(resolved) != len(set(resolved)):
        raise ConfigurationError("fixtures contains duplicate resolved paths")
    for fixture in resolved:
        _fixture(fixture)
    values["fixtures"] = resolved

    bounds = {
        "prompt_tokens": (1, 32768),
        "generation_tokens": (1, 4096),
        "threads": (1, 4096),
        "gpu_layers": (-1, 4096),
        "batch_sizes": (1, 32768),
        "ubatch_sizes": (1, 32768),
    }
    for field, (minimum, maximum) in bounds.items():
        dimension = values.get(field, RequestConfig.__dataclass_fields__[field].default)
        if not isinstance(dimension, (list, tuple)) or not dimension:
            raise ConfigurationError(f"{field} must be a nonempty list of integers")
        normalized = tuple(_integer(value, field, minimum, maximum) for value in dimension)
        if len(normalized) != len(set(normalized)):
            raise ConfigurationError(f"{field} contains duplicate settings")
        values[field] = normalized

    for field, minimum, maximum in (
        ("context_size", 3, 32768),
        ("repetitions", 1, 1000),
        ("warmups", 1, 10),
        ("seed", 0, 2**32 - 1),
        ("shuffle_seed", 0, 2**32 - 1),
    ):
        values[field] = _integer(
            values.get(field, RequestConfig.__dataclass_fields__[field].default),
            field,
            minimum,
            maximum,
        )
    for field in ("timeout_s", "startup_timeout_s"):
        value = values.get(field, RequestConfig.__dataclass_fields__[field].default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 86400:
            raise ConfigurationError(f"{field} must be finite and in (0, 86400]")
        values[field] = float(value)
    for field in ("ignore_eos", "cache_prompt"):
        value = values.get(field, RequestConfig.__dataclass_fields__[field].default)
        if type(value) is not bool:
            raise ConfigurationError(f"{field} must be a boolean")
        values[field] = value
    if values["cache_prompt"]:
        raise ConfigurationError("cache_prompt must be false; prefix reuse is not supported")
    if values.get("flash_attention", "auto") not in ("on", "off", "auto"):
        raise ConfigurationError("flash_attention must be on, off, or auto (quote YAML strings)")
    if values.get("mode", "request_warm") != "request_warm":
        raise ConfigurationError("Supported mode: request_warm")

    config = RequestConfig(**values)
    if _matrix_size(config) > MAX_REQUEST_CASES:
        raise ConfigurationError("Experiment matrix exceeds 128 request cases; split the suite")
    if any(ubatch > batch for ubatch in config.ubatch_sizes for batch in config.batch_sizes):
        raise ConfigurationError("Every ubatch_size must be <= every batch_size")
    # Leave one position for the server's generation boundary; never rely on context shifting.
    if max(config.prompt_tokens) + max(config.generation_tokens) + 1 > config.context_size:
        raise ConfigurationError("prompt_tokens + generation_tokens + 1 must fit context_size")
    return config
